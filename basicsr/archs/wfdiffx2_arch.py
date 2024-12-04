import math
import torch
from torch import nn
from basicsr.utils.registry import ARCH_REGISTRY
from .Padiff_arch.wavelet import DWT, IWT
from .Padiff_arch.inr_arch import INR
from .Padiff_arch.diffx2_arch import GaussianDiffusionx2

from .Padiff_arch.cfc_arch import CFC
from .Padiff_arch.dft_arch import DFT,DFTHL,cross_attention,DFTHL1


@ARCH_REGISTRY.register()
class WfDiffx2(nn.Module): 
    def __init__(
            self, 
            in_channel=6,
            out_channel=3,
            inner_channel=32,
            norm_groups=32,
            with_time_emb=True,
            schedule_opt=None,
            sample_proc = 'ddim',
            local_ensemble=True, 
            feat_unfold=True, 
            cell_decode=True,
            ppg_input_channels=3,
    ):
        super().__init__()
        self.denoiser1= GaussianDiffusionx2(
            in_channel=in_channel,
            out_channel=out_channel,
            inner_channel=inner_channel,
            norm_groups=norm_groups,
            with_time_emb=with_time_emb,
            schedule_opt=schedule_opt,
            sample_proc = 'sample_proc'
        )
        self.denoiser2 = GaussianDiffusionx2(
            in_channel=in_channel,
            out_channel=out_channel,
            inner_channel=inner_channel,
            norm_groups=norm_groups,
            with_time_emb=with_time_emb,
            schedule_opt=schedule_opt,
            sample_proc = 'sample_proc'
        )
        self.init_predictor = DFTHL1()
      
        self.cfc = CFC(dim=int(3), num_heads=1)
        
    def forward(self, condition,gt):

        dwt,idwt= DWT(),IWT()
        
        x_,AA,HH = self.init_predictor(condition)
        input_img = x_[:, :3, :, :]
        n, c, h, w = input_img.shape
        
       
        input_dwt = dwt(x_)
        input_LL, input_high0 = input_dwt[:n, ...], input_dwt[n:, ...]

        x_HH,x_LL=self.cfc(input_LL, input_high0)

        if gt != None :
            gt_dwt = dwt(gt)
            gt_LL, gt_high0 = gt_dwt[:n, ...], gt_dwt[n:, ...]
            residual_LL = gt_LL - input_LL
            residual_high0 = gt_high0 - input_high0     
        else :
            residual_LL = None
            residual_high0 = None

        noisell,x__LL = self.denoiser1(input_LL,  residual_LL,x_LL)
        noisehigh,x__high0 = self.denoiser2(input_high0,residual_high0,x_HH)

        return x_, noisell,x__LL,noisehigh,x__high0,AA,HH
        

    
