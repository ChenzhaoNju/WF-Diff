import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
from einops import rearrange
#from wavelet import DWT, IWT
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


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # Squeeze操作：全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation操作：全连接层
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0)
        
        # Sigmoid激活函数，用于输出注意力权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze操作
        avg_out = self.global_avg_pool(x)
        
        # Excitation操作
        x = self.fc1(avg_out)
        x = F.relu(x)
        x = self.fc2(x)
        
        # 得到通道注意力权重
        attention = self.sigmoid(x)
        
        # 通道加权
        return x * attention

class Wide_Transformer(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Wide_Transformer, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 4, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 4, dim * 4, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.channel=ChannelAttention(dim)

    def forward(self, x):
        res=x
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v,l = qkv.chunk(4, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        l = rearrange(l, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        L=self.channel(l)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out=L+out+res

        out = self.project_out(out)+L
        return out
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, 2 * (i // 2) / np.float32(d_model))
    return pos * angle_rates



def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[:, np.newaxis, :]
    return torch.tensor(pos_encoding, dtype=torch.float32)

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def nonlinearity(x):
    return x*torch.sigmoid(x)

class DTB(nn.Module):

    def __init__(self, dim, num_heads, ffn_factor, bias, LayerNorm_type):
        super(DTB, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_factor, bias)

    def forward(self, x):
       
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


# handle multiple input 处理多个input
class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

#####################################Diffusion Transformer DFT################################
class DFT(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 heads=[1, 2, 4, 8],
                 ffn_factor = 4.0,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False
                 ):

        super(DFT, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = MySequential(*[
            DTB(dim=dim, num_heads=heads[0], ffn_factor=ffn_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = MySequential(*[
            DTB(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = MySequential(*[
            DTB(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = MySequential(*[
            DTB(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = MySequential(*[
            DTB(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = MySequential(*[
            DTB(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = MySequential(*[
            DTB(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, t):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1,_ = self.encoder_level1(inp_enc_level1, t)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2,_ = self.encoder_level2(inp_enc_level2, t)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3,_ = self.encoder_level3(inp_enc_level3, t)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent,_ = self.latent(inp_enc_level4, t)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3,_ = self.decoder_level3(inp_dec_level3, t)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2,_ = self.decoder_level2(inp_dec_level2, t)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1,_ = self.decoder_level1(inp_dec_level1, t)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img[:,-3:,:,:]

        return out_dec_level1



class DFTHL(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 heads=[1, 2, 4, 8],
                 ffn_factor = 4.0,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False
                 ):

        super(DFTHL, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed1 = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 =  FRDB(nChannels=dim) 
        #self.encoder_level1 = MySequential(*[
            #DTB(dim=dim, num_heads=heads[0], ffn_factor=ffn_factor, bias=bias,
                             #LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level11 = MySequential(*[
            DTB(dim=dim, num_heads=heads[0], ffn_factor=ffn_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
         



        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.down1_21 = Downsample(dim)
        
        self.encoder_level2 = FRDB(nChannels=dim * 2 ** 1)
        #self.encoder_level2 = MySequential(*[
            #DTB(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_factor=ffn_factor,
                             #bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level21 = MySequential(*[
            DTB(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.down2_31 = Downsample(int(dim * 2 ** 1))

        self.encoder_level3 = FRDB(nChannels=dim * 2 ** 2)
        #self.encoder_level3 = MySequential(*[
            #DTB(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_factor=ffn_factor,
                             #bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.encoder_level31 = MySequential(*[
            DTB(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.down3_41 = Downsample(int(dim * 2 ** 2))

        self.latent = FRDB(nChannels=dim * 2 ** 3)
        #self.latent = MySequential(*[
            #DTB(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_factor=ffn_factor,
                             #bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.latent1 = MySequential(*[
            DTB(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.cross_attention0 = cross_attention(dim=int(dim * 2 ** 3), num_heads=8)

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.up4_31 = Upsample(int(dim * 2 ** 3))

        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.reduce_chan_level31 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)

        self.decoder_level3 = FRDB(nChannels=dim * 2 ** 2)
        #self.decoder_level3 = MySequential(*[
            #DTB(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_factor=ffn_factor,
                             #bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.decoder_level31 = MySequential(*[
            DTB(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.up3_21 = Upsample(int(dim * 2 ** 2))

        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level21 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)

        self.decoder_level2 = FRDB(nChannels=dim * 2 ** 1)
        #self.decoder_level2 = MySequential(*[
            #DTB(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_factor=ffn_factor,
                             #bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.decoder_level21 = MySequential(*[
            DTB(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.up2_11 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = FRDB(nChannels=dim * 2 ** 1)
        #self.decoder_level1 = MySequential(*[
            #DTB(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_factor=ffn_factor,
                             #bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level11 = MySequential(*[
            DTB(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output1 = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.upH1 = Upsample(int(dim * 2 ** 3))
        self.upL1 = Upsample(int(dim * 2 ** 3))
        self.upL2 = Upsample(int(dim * 2 ** 2))
        self.upH2 = Upsample(int(dim * 2 ** 2))

        

    def forward(self, inp_img):

        dwt,idwt= DWT(),IWT()

        input_img = inp_img[:, :3, :, :]
        n, c, h, w = input_img.shape
    
        input_dwt = dwt(input_img)
        input_LL, input_high0 = input_dwt[:n, ...], input_dwt[n:, ...]


        inp_enc_level1 = self.patch_embed(input_LL)
        #out_enc_level1,_ = self.encoder_level1(inp_enc_level1, t)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        #out_enc_level2,_ = self.encoder_level2(inp_enc_level2, t)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        #out_enc_level3,_ = self.encoder_level3(inp_enc_level3, t)
        
        
        
        # out_enc_level3 = self.encoder_level3(inp_enc_level3)
        # inp_enc_level4 = self.down3_4(out_enc_level3)


        inp_enc_level11 = self.patch_embed1(input_high0)
        out_enc_level11 = self.encoder_level11(inp_enc_level11)
        inp_enc_level21 = self.down1_21(out_enc_level11)
        out_enc_level21 = self.encoder_level21(inp_enc_level21)
        inp_enc_level31 = self.down2_31(out_enc_level21)


        # out_enc_level31 = self.encoder_level31(inp_enc_level31)
        # inp_enc_level41 = self.down3_41(out_enc_level31) 
        # x_HH,x_LL = self.cross_attention0(inp_enc_level4, inp_enc_level41)
        # x_LL1 = self.upL1(x_LL)
        # x_HH1 = self.upH1(x_HH)

        x_HH,x_LL = self.cross_attention0(inp_enc_level3, inp_enc_level31)
        x_LL1 = self.upL1(x_LL)
        x_HH1 = self.upH1(x_HH)



        latent = self.latent(inp_enc_level3)
        #latent,_ = self.latent(inp_enc_level4, t)
        latent1 = self.latent1(inp_enc_level31)
        #print(latent.shape)
        #print(latent1.shape)
        #print( x_HH_LH.shape)
        inp_dec_level3 = self.up4_3(latent+x_LL)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level2], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        #out_dec_level3,_ = self.decoder_level3(inp_dec_level3, t)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3+x_LL1)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2= self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1)

        x_HH=torch.cat((x_HH,x_HH,x_HH),dim=0)
        inp_dec_level31 = self.up4_31(latent1+x_HH)
        inp_dec_level31 = torch.cat([inp_dec_level31, out_enc_level31], 1)
        inp_dec_level31 = self.reduce_chan_level31(inp_dec_level31)
        out_dec_level31= self.decoder_level31(inp_dec_level31)
        x_HH1=torch.cat((x_HH1,x_HH1,x_HH1),dim=0)
        inp_dec_level21 = self.up3_21(out_dec_level31+x_HH1)
        inp_dec_level21 = torch.cat([inp_dec_level21, out_enc_level21], 1)
        inp_dec_level21 = self.reduce_chan_level21(inp_dec_level21)
        out_dec_level21 = self.decoder_level21(inp_dec_level21)
        inp_dec_level11 = self.up2_11(out_dec_level21)
        inp_dec_level11 = torch.cat([inp_dec_level11, out_enc_level11], 1)
        out_dec_level11 = self.decoder_level11(inp_dec_level11)

        #### For Dual-Pixel Defocus Deblurring Task ####
        
        ###########################
       
        out_dec_level11 = self.output(out_dec_level11)

        out_dec_level=idwt(torch.cat((out_dec_level1,out_dec_level11),dim=0))

        return out_dec_level,out_dec_level1,out_dec_level11
    


class DFTHL1(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 heads=[1, 2, 4, 8],
                 ffn_factor = 4.0,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False
                 ):

        super(DFTHL1, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed1 = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 =  FRDB(nChannels=dim) 
        #self.encoder_level1 = MySequential(*[
            #DTB(dim=dim, num_heads=heads[0], ffn_factor=ffn_factor, bias=bias,
                             #LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level11 = MySequential(*[
            DTB(dim=dim, num_heads=heads[0], ffn_factor=ffn_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
         



        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.down1_21 = Downsample(dim)
        
        self.encoder_level2 = FRDB(nChannels=dim * 2 ** 1)
        #self.encoder_level2 = MySequential(*[
            #DTB(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_factor=ffn_factor,
                             #bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level21 = MySequential(*[
            DTB(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.down2_31 = Downsample(int(dim * 2 ** 1))

        self.encoder_level3 = FRDB(nChannels=dim * 2 ** 2)
        #self.encoder_level3 = MySequential(*[
            #DTB(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_factor=ffn_factor,
                             #bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.encoder_level31 = MySequential(*[
            DTB(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.down3_41 = Downsample(int(dim * 2 ** 2))

        self.latent = FRDB(nChannels=dim * 2 ** 2)
        #self.latent = MySequential(*[
            #DTB(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_factor=ffn_factor,
                             #bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.latent1 = MySequential(*[
            DTB(dim=int(dim * 2 ** 2), num_heads=heads[3], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.cross_attention0 = cross_attention(dim=int(dim * 2 ** 2), num_heads=8)

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.up4_31 = Upsample(int(dim * 2 ** 3))

        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.reduce_chan_level31 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)

        self.decoder_level3 = FRDB(nChannels=dim * 2 ** 2)
        #self.decoder_level3 = MySequential(*[
            #DTB(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_factor=ffn_factor,
                             #bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.decoder_level31 = MySequential(*[
            DTB(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.up3_21 = Upsample(int(dim * 2 ** 2))

        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level21 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)

        self.decoder_level2 = FRDB(nChannels=dim * 2 ** 1)
        #self.decoder_level2 = MySequential(*[
            #DTB(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_factor=ffn_factor,
                             #bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.decoder_level21 = MySequential(*[
            DTB(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.up2_11 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = FRDB(nChannels=dim * 2 ** 1)
        #self.decoder_level1 = MySequential(*[
            #DTB(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_factor=ffn_factor,
                             #bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level11 = MySequential(*[
            DTB(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_factor=ffn_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output1 = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.upH1 = Upsample(int(dim * 2 ** 2))
        self.upL1 = Upsample(int(dim * 2 ** 2))
        # self.upL2 = Upsample(int(dim * 2 ** 2))
        # self.upH2 = Upsample(int(dim * 2 ** 2))

        

    def forward(self, inp_img):

        dwt,idwt= DWT(),IWT()

        input_img = inp_img[:, :3, :, :]
        n, c, h, w = input_img.shape
    
        input_dwt = dwt(input_img)
        input_LL, input_high0 = input_dwt[:n, ...], input_dwt[n:, ...]


        inp_enc_level1 = self.patch_embed(input_LL)
        #out_enc_level1,_ = self.encoder_level1(inp_enc_level1, t)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        #out_enc_level2,_ = self.encoder_level2(inp_enc_level2, t)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        #out_enc_level3,_ = self.encoder_level3(inp_enc_level3, t)
        
        
        
        # out_enc_level3 = self.encoder_level3(inp_enc_level3)
        # inp_enc_level4 = self.down3_4(out_enc_level3)


        inp_enc_level11 = self.patch_embed1(input_high0)
        out_enc_level11 = self.encoder_level11(inp_enc_level11)
        inp_enc_level21 = self.down1_21(out_enc_level11)
        out_enc_level21 = self.encoder_level21(inp_enc_level21)
        inp_enc_level31 = self.down2_31(out_enc_level21)


        # out_enc_level31 = self.encoder_level31(inp_enc_level31)
        # inp_enc_level41 = self.down3_41(out_enc_level31) 
        # x_HH,x_LL = self.cross_attention0(inp_enc_level4, inp_enc_level41)
        # x_LL1 = self.upL1(x_LL)
        # x_HH1 = self.upH1(x_HH)

        x_HH,x_LL = self.cross_attention0(inp_enc_level3, inp_enc_level31)
        x_LL1 = self.upL1(x_LL)
        x_HH1 = self.upH1(x_HH)



        latent = self.latent(inp_enc_level3)
        #latent,_ = self.latent(inp_enc_level4, t)
        latent1 = self.latent1(inp_enc_level31)
        #print(latent.shape)
        #print(latent1.shape)
        #print( x_HH_LH.shape)
        # inp_dec_level3 = self.up4_3(latent+x_LL)
        # inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level2], 1)
        # inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        # #out_dec_level3,_ = self.decoder_level3(inp_dec_level3, t)
        # out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(latent+x_LL)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2= self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2+x_LL1)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1)

        x_HH=torch.cat((x_HH,x_HH,x_HH),dim=0)
        # inp_dec_level31 = self.up4_31(latent1+x_HH)
        # inp_dec_level31 = torch.cat([inp_dec_level31, out_enc_level31], 1)
        # inp_dec_level31 = self.reduce_chan_level31(inp_dec_level31)
        # out_dec_level31= self.decoder_level31(inp_dec_level31)
        
        inp_dec_level21 = self.up3_21(latent1+x_HH)
        inp_dec_level21 = torch.cat([inp_dec_level21, out_enc_level21], 1)
        inp_dec_level21 = self.reduce_chan_level21(inp_dec_level21)
        out_dec_level21 = self.decoder_level21(inp_dec_level21)
        x_HH1=torch.cat((x_HH1,x_HH1,x_HH1),dim=0)
        inp_dec_level11 = self.up2_11(out_dec_level21+x_HH1)
        inp_dec_level11 = torch.cat([inp_dec_level11, out_enc_level11], 1)
        out_dec_level11 = self.decoder_level11(inp_dec_level11)

        #### For Dual-Pixel Defocus Deblurring Task ####
        
        ###########################
       
        out_dec_level11 = self.output(out_dec_level11)

        out_dec_level=idwt(torch.cat((out_dec_level1,out_dec_level11),dim=0))

        return out_dec_level,out_dec_level1,out_dec_level11
    

class cross_attention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super(cross_attention, self).__init__()
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

        ctx_layerl = torch.matmul(attention_probs, value_layerl)
        ctx_layerl = ctx_layerl.permute(0, 2, 1, 3).contiguous()

        return ctx_layerh,ctx_layerl

class make_fdense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=1):
        super(make_fdense, self).__init__()
        #self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              #bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False),nn.BatchNorm2d(growthRate)
        )
        self.bat = nn.BatchNorm2d(growthRate),
        self.leaky=nn.LeakyReLU(0.1,inplace=True)

    def forward(self, x):
        out = self.leaky(self.conv(x))
        out = torch.cat((x, out), 1)
        return out
    
class FRDB(nn.Module):
    def __init__(self, nChannels, nDenselayer=1, growthRate=32):
        super(FRDB, self).__init__()
        nChannels_1 = nChannels
        nChannels_2 = nChannels
        modules1 = []
        for i in range(nDenselayer):
            modules1.append(make_fdense(nChannels_1, growthRate))
            nChannels_1 += growthRate
        self.dense_layers1 = nn.Sequential(*modules1)
        modules2 = []
        for i in range(nDenselayer):
            modules2.append(make_fdense(nChannels_2, growthRate))
            nChannels_2 += growthRate
        self.dense_layers2 = nn.Sequential(*modules2)
        self.conv_1 = nn.Conv2d(nChannels_1, nChannels, kernel_size=1, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(nChannels_2, nChannels, kernel_size=1, padding=0, bias=False)
        self.SRDB=SRDB(nChannels)
        #self.patch_embed = PatchEmbed(img_size=224, patch_size=7, stride=4, in_chans=nChannels,
                                              #embed_dim=embed_dims[0])


    def forward(self, x):
        x=self.SRDB(x)
        _, _, H, W = x.shape
        #print(x.shape)
        x_freq = torch.fft.rfft2(x, norm='backward')
        #print(x_freq.shape)
        mag = torch.abs(x_freq)
        #print(mag.shape)
        pha = torch.angle(x_freq)
        mag = self.dense_layers1(mag)
        #print(mag.shape)
        mag = self.conv_1(mag)
        #print(mag.shape)
        pha = self.dense_layers2(pha)
        pha = self.conv_2(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        out = out + x
        return out


class SRDB(nn.Module):
    def __init__(self, nChannels, growthRate=64):
        super(SRDB, self).__init__()
        nChannels_ = nChannels
        modules1 = []
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=1, padding=(1 - 1) // 2,
                              bias=False)
        self.conv2 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=(3 - 1) // 2,
                              bias=False)
        self.conv3 = nn.Conv2d(nChannels, growthRate, kernel_size=5, padding=(5 - 1) // 2,
                              bias=False)
        
        #self.conv11 = nn.Conv2d(nChannels, growthRate, kernel_size=1, padding=(1 - 1) // 2,
        #                      bias=False)
        #self.conv22 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=(3 - 1) // 2,
        #                      bias=False)
        #self.conv33 = nn.Conv2d(nChannels, growthRate, kernel_size=5, padding=(5 - 1) // 2,
        #                      bias=False)
        
        #self.conv4 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=(3 - 1) // 2,
         #                     bias=False)
        #self.conv5 = nn.Conv2d(nChannels, growthRate, kernel_size=5, padding=(5 - 1) // 2,
        #                      bias=False)
        self.conv6 = nn.Conv2d(growthRate*3, nChannels, kernel_size=1, padding=(1 - 1) // 2,
                              bias=False)
        self.leaky1=nn.LeakyReLU(0.1,inplace=True)
        self.leaky2=nn.LeakyReLU(0.1,inplace=True)
        self.leaky3=nn.LeakyReLU(0.1,inplace=True)
        #self.bat1 = nn.BatchNorm2d(nChannels),
        #self.bat2 = nn.BatchNorm2d(nChannels),
        #self.bat3 = nn.BatchNorm2d(nChannels),
        #self.bat4 = nn.BatchNorm2d(nChannels),
        #self.bat5 = nn.BatchNorm2d(nChannels),
        #self.patch_embed = PatchEmbed(img_size=224, patch_size=7, stride=4, in_chans=nChannels,
                                              #embed_dim=embed_dims[0])


    def forward(self, x):
        #x_1=self.bat1(self.conv1(x))
        x_1= self.leaky1(self.conv1(x))
        x_2= self.leaky2(self.conv2(x))
        x_3= self.leaky3(self.conv3(x))
        x_0=torch.cat((x_1,x_2,x_3),dim=1)
        #print(x_0.shape)

        #x_11=x_1+x_3
        #x_22=x_1+x_3+x_2
        #x_33=x_2+x_3

        #x_111= self.conv11(x_11)
        #x_222= self.conv22(x_22)
        #x_333= self.conv33(x_33)

        #x111=x_111*x_222+x_111
        #x333=x_222*x_333+x_333

        #x_o1= self.conv4(x111)
        #x_02= self.conv5(x333)

        #x_0=x_o1+x_02+x_1+x_3

        #x_0=self.conv6(x_0)
        #x_0=x_111+x+x_222+x_333
        x_0=self.conv6(x_0)

        out = x_0 + x
        return out
