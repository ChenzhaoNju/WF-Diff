import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange



##########################################################################
## Layer Norm

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
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        self.x1_Conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.x1_Conv5 = nn.Conv2d(dim, dim, kernel_size=5,padding=2, bias=bias)
        self.x2_Conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.x2_Conv5 = nn.Conv2d(dim, dim, kernel_size=5,padding=2, bias=bias)
        self.x3_Conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.x3_Conv5 = nn.Conv2d(dim, dim, kernel_size=5,padding=2, bias=bias)
        self.x4_Conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.x4_Conv5 = nn.Conv2d(dim, dim, kernel_size=5,padding=2, bias=bias)
        self.x5_Conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.x5_Conv5 = nn.Conv2d(dim, dim, kernel_size=5,padding=2, bias=bias)

        self.outCon = nn.Conv2d(2*dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):

        x1 = self.x1_Conv5(self.x1_Conv1(x))
        x2 = self.x2_Conv5(self.x2_Conv1(x))
        x3 = self.x3_Conv5(self.x3_Conv1(x))

        x1 = x2 + self.x4_Conv5(self.x4_Conv1(F.gelu(x1) * x2))
        x3 = x2 + self.x5_Conv5(self.x5_Conv1(F.gelu(x3) * x2))

        return self.outCon(torch.concat([x1,x3],dim=1))


# Multi-DConv Head Transposed Self-Attention (MDTA)
class PA_MSA(nn.Module):
    def __init__(self, dim, num_heads, bias,t_dim = 3):
        super(PA_MSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        
        self.q_T = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv_T = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.q_concat = nn.Conv2d(2*dim, dim, kernel_size=1, bias=bias)
        
        self.k_T = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_dwconv_T = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.k_concat = nn.Conv2d(2*dim, dim, kernel_size=1, bias=bias)

        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)


    def forward(self, x):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        
        k = self.k_dwconv(self.k(x))
       
        v = self.v_dwconv(self.v(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
       
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
       
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # out = out + x

        # out = self.project_out(out)
        return out

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, with_PPU=False):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.atten = PA_MSA(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.with_PPU = with_PPU

    def forward(self, x, time):
        x = x + self.atten(self.norm1(x)+time)
        x = x + self.ffn(self.norm2(x)+time)
        return x

class Swish(nn.Module):
    """
    ### Swish actiavation function
    $$x \cdot \sigma(x)$$
    """

    def forward(self, x):
        return x * torch.sigmoid(x)
    


class ResidualBlock(nn.Module):
    """
    ### Residual block
    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 dropout: float = 0.1, is_noise: bool = True):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        * `dropout` is the dropout rate
        """
        super().__init__()
        # Group normalization and the first convolution layer
        self.is_noise = is_noise
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Group normalization and the second convolution layer

        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        if self.is_noise:
            self.time_emb = nn.Linear(time_channels, out_channels)
            self.time_act = Swish()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, time: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        h = self.conv1(self.act1(x))
        # Add time embeddings
        h += time
        # Second convolution layer
        h = self.conv2(self.dropout(self.act2(h)))
        # Add the shortcut connection and return
        return h + self.shortcut(x)