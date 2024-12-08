import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from .unet_block.trans_block_eca import ResidualBlock,TransformerBlock
from .unet_block.cross_attention_module import CFC


# model
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class ResnetBloc_eca(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=32, dropout=0, with_attn=False,with_PPU = True):
        super().__init__()
        self.with_attn = with_attn
        if with_attn:
            #self.attn = ResidualBlock(dim, dim, dim_out, is_noise=True)
            self.attn = TransformerBlock(dim=int(dim), num_heads=dim, ffn_expansion_factor=2.66,
                               bias=False, LayerNorm_type='WithBias',with_PPU= with_PPU)
        
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim_out)
        )
    def forward(self, x, time_emb, f_out):
        # print(x.shape)
        # print( f_out.shape)
        x=x+f_out
        
        time = self.mlp(time_emb).unsqueeze(2).unsqueeze(3)
        if self.with_attn :
            x = self.attn(x, time)
        return x

class Encoder(nn.Module):
    def __init__(
            self,
            in_channel=6,
            inner_channel=32,
            norm_groups=32,
    ):
        super().__init__()

        dim = inner_channel
        time_dim = inner_channel
        # x
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, dim, kernel_size=3, stride=1, padding=1, bias=False))
            # ,nn.PixelUnshuffle(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1) // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 2) // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2))
        
        self.conv1_t = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1, bias=False))
            # ,nn.PixelUnshuffle(2))
        self.conv2_t = nn.Sequential(
            nn.Conv2d(3, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2))
        self.conv3_t = nn.Sequential(
            nn.Conv2d(3, int(dim * 2 ** 1) // 8, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2),nn.PixelUnshuffle(2))
        self.conv4_t = nn.Sequential(
            nn.Conv2d(3, int(dim * 2 ** 2) // 32, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2),nn.PixelUnshuffle(2),nn.PixelUnshuffle(2))
        
        self.cam1 = CFC(dim,dim)
        self.cam2 = CFC(dim * 2 ** 1, dim * 2 ** 1)
        self.cam3 = CFC(dim * 2 ** 2, dim * 2 ** 2)
        self.cam4 = CFC(dim * 2 ** 3, dim * 2 ** 3)
        

        self.block1 = ResnetBloc_eca(dim=dim, dim_out=dim, time_emb_dim=time_dim, norm_groups=norm_groups,
                                     with_attn=True)
        self.block2 = ResnetBloc_eca(dim=dim * 2 ** 1, dim_out=dim * 2 ** 1, time_emb_dim=time_dim,
                                     norm_groups=norm_groups, with_attn=True)
        self.block3 = ResnetBloc_eca(dim=dim * 2 ** 2, dim_out=dim * 2 ** 2, time_emb_dim=time_dim,
                                     norm_groups=norm_groups, with_attn=True)
        self.block4 = ResnetBloc_eca(dim=dim * 2 ** 3, dim_out=dim * 2 ** 3, time_emb_dim=time_dim,
                                     norm_groups=norm_groups, with_attn=True,with_PPU=True)

        self.conv_up3 = nn.Sequential(
            nn.Conv2d((dim * 2 ** 3), (dim * 2 ** 3) * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2))

        self.conv_up2 = nn.Sequential(
            nn.Conv2d((dim * 2 ** 2), (dim * 2 ** 2) * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2))
        self.conv_up1 = nn.Sequential(
            nn.Conv2d((dim * 2 ** 1), (dim * 2 ** 1) * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2))

        self.conv_cat3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=False)
        self.conv_cat2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=False)
        self.conv_cat1 = nn.Conv2d(int(dim * 2 ** 1), int(dim), kernel_size=1, bias=False)

        self.decoder_block3 = ResnetBloc_eca(dim=dim * 2 ** 2, dim_out=dim * 2 ** 2, time_emb_dim=time_dim,
                                             norm_groups=norm_groups, with_attn=True)
        self.decoder_block2 = ResnetBloc_eca(dim=dim * 2 ** 1, dim_out=dim * 2 ** 1, time_emb_dim=time_dim,
                                             norm_groups=norm_groups, with_attn=True)
        self.decoder_block1 = ResnetBloc_eca(dim=dim, dim_out=dim, time_emb_dim=time_dim,
                                             norm_groups=norm_groups, with_attn=True)

    def forward(self, x, t, p_t):
        # 1
        x = self.conv1(x)
        # print(p_t.shape)
        # print(x.shape)
        f_out1 = self.conv1_t(p_t)
        # print(f_out1.shape)
        # print(x.shape)
        f_out1 = self.cam1(f_out1,x)
        x1 = self.block1(x, t, f_out1)

        # 2
        x2 = self.conv2(x1)
        f_out2 = self.conv2_t(p_t)
        f_out2 = self.cam2(f_out2,x2)
        x2 = self.block2(x2, t, f_out2)

        # 3
        x3 = self.conv3(x2)
        f_out3 = self.conv3_t(p_t)
        f_out3 = self.cam3(f_out3,x3)
        x3 = self.block3(x3, t, f_out3)

        # 4
        # x4 = self.conv4(x3)
        # f_out4 = self.conv4_t(p_t)
        # x4 = self.block4(x4, t, f_out4)

        # de_level3 = self.conv_up3(x4)
        # de_level3 = torch.cat([de_level3, x3], 1)
        # de_level3 = self.conv_cat3(de_level3)
        # de_level3 = self.decoder_block3(de_level3, t, f_out3)

        de_level2 = self.conv_up2(x3)
        de_level2 = torch.cat([de_level2, x2], 1)
        de_level2 = self.conv_cat2(de_level2)
        de_level2 = self.decoder_block2(de_level2, t, f_out2)

        de_level1 = self.conv_up1(de_level2)
        de_level1 = torch.cat([de_level1, x1], 1)
        de_level1 = self.conv_cat1(de_level1)
        mid_feat = self.decoder_block1(de_level1, t, f_out1)

        return mid_feat

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.eps = 1e-5

    def forward(self, x, y):
        mean_x, mean_y = torch.mean(x, dim=(2, 3), keepdim=True), torch.mean(y, dim=(2, 3), keepdim=True)
        std_x, std_y = torch.std(x, dim=(2, 3), keepdim=True) + self.eps, torch.std(y, dim=(2, 3), keepdim=True) + self.eps
        return std_y * (x - mean_x) / std_x + mean_y

class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        with_time_emb=True
    ):
        super().__init__()

        if with_time_emb:
            time_dim = inner_channel
            self.time_mlp = nn.Sequential(
                TimeEmbedding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            time_dim = None
            self.time_mlp = None

        dim = inner_channel

        self.encoder_water = Encoder(in_channel=in_channel, inner_channel=inner_channel, norm_groups=norm_groups)

        self.refine = ResnetBloc_eca(dim=dim*2**1, dim_out=dim*2**1, time_emb_dim=time_dim, norm_groups=norm_groups, with_attn=False)
        self.de_predict = nn.Sequential(nn.Conv2d(dim, out_channel, kernel_size=1, stride=1))


    def forward(self, x, time, p):

        t = self.time_mlp(time) if self.time_mlp is not None else None

        mid_feat = self.encoder_water(x, t, p)
        return self.de_predict(mid_feat)
