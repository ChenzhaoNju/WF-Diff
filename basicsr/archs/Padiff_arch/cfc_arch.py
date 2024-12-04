from tkinter import Image
import torch
from torch import nn
import numpy as np
from torchvision.transforms import ToTensor
from PIL import ImageFilter
from .unet_block.ConditionNet import ConditionNet
from basicsr.utils.registry import ARCH_REGISTRY
from torchvision.transforms import ToPILImage

import math
def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    # ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    # if img_np.shape[0] == 1:
    #     ar = ar[0]
    # else:
    #     assert img_np.shape[0] == 3, img_np.shape
    #     ar = ar.transpose(1, 2, 0)

    # return Image.fromarray(ar)
    # return ToPILImage()(img_np)
    
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    return ToPILImage()(img_np)
    

def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    # return img_var.detach().cpu().numpy()[0]
    return img_var.detach().cpu().numpy().transpose(1, 2, 0) 

def get_A(x):

    processed_images = []

    for i in range(x.size(0)):
        img_np = torch_to_np(x[i])
        img_pil = np_to_pil(img_np)
        
        h, w = img_pil.size
        windows = (h + w) / 2
        
        A_pil = img_pil.filter(ImageFilter.GaussianBlur(windows))
        
        A_tensor = ToTensor()(A_pil).to(x.device)
        processed_images.append(A_tensor)
    
    return torch.stack(processed_images, dim=0)

class TNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.aNet = ConditionNet(support_size=input_channels)
        self.final = nn.Conv2d(128, output_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        a = self.final(self.aNet(x))
        return a
 
@ARCH_REGISTRY.register()
class PPG(nn.Module):
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.tNet = TNet(input_channels,1)
        self.aNet = TNet(input_channels,3)

    def forward(self, x): 
        a = self.aNet(get_A(x)+x)
        t = self.tNet(x)
        return a, t




class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
    
@ARCH_REGISTRY.register()
class CFC(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super(CFC, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (dim, num_heads)
            )
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)

        self.query = Depth_conv(in_ch=dim, out_ch=dim)
        self.key = Depth_conv(in_ch=dim, out_ch=dim)
        self.valueh = Depth_conv(in_ch=dim, out_ch=dim)
        self.valuel = Depth_conv(in_ch=dim, out_ch=dim)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        '''
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.attention_head_size,
        )
        print(new_x_shape)
        x = x.view(*new_x_shape)
        '''
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, ctx):
        n, c, h, w = hidden_states.shape
        ctx1 = ctx[:n, ...]
        ctx2 =  ctx[n:n+n, ...]
        ctx3 =  ctx[n+n:, ...]
        ctx=ctx1+ctx2+ctx3
        
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(ctx)
        mixed_value_layerh = self.valueh(ctx)
        mixed_value_layerl = self.valuel(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layerh = self.transpose_for_scores(mixed_value_layerh)
        value_layerl = self.transpose_for_scores(mixed_value_layerl)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        ctx_layerh = torch.matmul(attention_probs, value_layerh)
        ctx_layerh = ctx_layerh.permute(0, 2, 1, 3).contiguous()
        ctx_layerh = ctx_layerh.repeat(3, 1, 1, 1)

        ctx_layerl = torch.matmul(attention_probs, value_layerl)
        ctx_layerl = ctx_layerl.permute(0, 2, 1, 3).contiguous()

        return ctx_layerh,ctx_layerl