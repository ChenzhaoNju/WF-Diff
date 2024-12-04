from functools import partial
import torch
import cv2
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F
from basicsr.models.sr_model import SRModel
#from wavelet import DWT, IWT
import torch
import torch.nn as nn

import numpy as np
from utils.diff_util import *
from utils.img_util import rgb_to_hsv_tensor


def Normalize(x):
    ymax = 255
    ymin = 0
    xmax = x.max()
    xmin = x.min()
    return (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin


def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)


# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = int(in_batch/(r**2)),in_channel, r * in_height, r * in_width
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


@MODEL_REGISTRY.register()
class WfdiffModel(SRModel):

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        self.eta = 0

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        if self.is_train:
            self.init_training_settings()


    def init_training_settings(self):
        self.net_g.train()
        # self.net_g.set_new_noise_schedule(self.opt["beta_schedule"],self.device)
        train_opt = self.opt['train']
        
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()
        
        
        if train_opt.get('Rec_loss'):
            self.Rec_loss = build_loss(train_opt['Rec_loss']).to(self.device)
        else:
            self.Rec_loss = None

        if train_opt.get('amp_loss'):
            self.amp_loss = build_loss(train_opt['amp_loss']).to(self.device)
        else:
            self.amp_loss = None
        
        if train_opt.get('high_loss'):
            self.high_loss = build_loss(train_opt['high_loss']).to(self.device)
        else:
            self.high_loss = None

        if train_opt.get('diff_loss'):
            self.diff_loss1 = build_loss(train_opt['diff_loss']).to(self.device)
            self.diff_loss2 = build_loss(train_opt['diff_loss']).to(self.device)
        else:
            self.diff_loss = None


        if self.Rec_loss is None and self.amp_loss is None and self.diff_loss is None:
            raise ValueError('Both pixel and perceptual losses are None.')
        
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        
        
    # TODO 改了
    def feed_data(self, data):
        self.pre = data['pre']
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        

    # TODO 改了
    def optimize_parameters(self, current_iter):
        
        self.optimizer_g.zero_grad()
        l_total = 0
        loss_dict = OrderedDict()
    
        # lq输入无力模型 得到a t
        # self.p_a ,self.p_t = self.net_ppg(self.lq)
        
        # 从inr获取得到 nrn_out
        # self.nrn_out = self.net_inr(self.lq)
        
        #self.noise, self.x_recon, self.p_a, self.p_t, self.nrn_out = self.net_g(self.lq,self.gt)
        self.x_, self.noisell,self.LLrecon,self.noisehigh,self.highrecon,self.AA,self.HH= self.net_g(self.lq,self.gt)
        # self.phy_out = self.gt * self.p_t + (1 - self.p_t) * self.p_a
        dwt,idwt= DWT(),IWT()
        n, c, h, w = self.gt.shape
        self.gt_dwt = dwt(self.gt)
        self.gt_LL, self.gt_high0 = self.gt_dwt[:n, ...], self.gt_dwt[n:, ...]


        # TODO LOSS 函数定义
        if self.Rec_loss:
            l_rec = self.Rec_loss(self.x_, self.gt)
            l_total += l_rec
            loss_dict['l_rec'] = l_rec
        
        if self.amp_loss:
            l_amp = self.amp_loss(self.AA, self.gt_LL)
            l_total += l_amp
            loss_dict['l_amp'] = l_amp
        if self.high_loss:
            l_high = self.high_loss(self.HH, self.gt_high0)
            l_total += l_high
            loss_dict['l_high'] = l_high

        if self.diff_loss1:
            l_diff1 = self.diff_loss1(self.noisell,self.LLrecon)
            l_total += l_diff1
            loss_dict['l_diff1'] = l_diff1
        if self.diff_loss2:
            l_diff2 = self.diff_loss2(self.noisehigh,self.highrecon)
            l_total += l_diff2
            loss_dict['l_diff2'] = l_diff2


        l_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)



    # TODO 改了
    def test(self):
        self.net_g.eval()
        dwt,idwt= DWT(),IWT()
        with torch.no_grad():
            #self.output,_, self.p_a, self.p_t, self.nrn_out = self.net_g(self.lq, None)
            self.out1, self.out2ll,_,self.out2high,_,self.AA,self.HH= self.net_g(self.lq, None)
            n, c, h, w = self.out1.shape                     
            self.out1dwt = dwt(self.out1)
            self.out1LL, self.out1high0 = self.out1dwt[:n, ...], self.out1dwt[n:, ...]
            self.finalLL=self.out1LL+self.out2ll 
            self.finalHH=self.out1high0+self.out2high
            self.output=idwt(torch.cat((self.finalLL,self.finalHH),dim=0))
        self.net_g.train()

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            # sr_img shape & pre shape  different 
            sr_resize = False
            pre_img = tensor2img([visuals['pre']])
            # pre_img = visuals['pre']
            if sr_img.shape != pre_img.shape:
                sr_img = cv2.resize(sr_img, (pre_img.shape[1],pre_img.shape[0]), interpolation=cv2.INTER_AREA)
                sr_resize = True

            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                # sr is resize
                if sr_resize:
                    gt_img = cv2.resize(gt_img, (pre_img.shape[1],pre_img.shape[0]), interpolation=cv2.INTER_AREA)
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            del self.pre
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        out_dict['pre'] = self.pre.detach().cpu()

        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict


