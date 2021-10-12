import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from swnn import swLinear
from swnn import swMHA
from swnn import swRelu
from swnn import swLayerNorm
import time

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        # self.norm = swLayerNorm(dim)
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim,):
        super().__init__()
        self.net = nn.Sequential(
            swLinear(dim, hidden_dim, bais=True),
            # nn.ReLU(),
            swRelu(),
            swLinear(hidden_dim, dim, bais=True),
        )
    def forward(self, x):
        st=time.perf_counter()
        xx=self.net(x)
        ed=time.perf_counter()
        # print(f'swLinear timing: {ed - st}')
        return xx


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, use_sw=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.use_sw = use_sw

        self.attend = nn.Softmax(dim = -1)

        if self.use_sw:
            # substitute multi-head attn layers
            self.mha = swMHA(dim, inner_dim * 3, heads, self.scale)
        else:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
            # only substitute fully connected layer
            # self.to_qkv = swLinear(dim, inner_dim * 3)

        self.to_out = nn.Sequential(
            swLinear(inner_dim, dim, bais=True),
            # nn.Linear(inner_dim, dim),
        )

    def forward(self, x):
        if self.use_sw:
            st=time.perf_counter()
            out = self.mha(x)
            ed=time.perf_counter()
            # print(f'swMHA timing: {ed - st}')
        else:
            b, n, _, h = *x.shape, self.heads
            qkv = self.to_qkv(x).chunk(3, dim = -1)

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

            attn = self.attend(dots)
            out = einsum('b h i j, b h j d -> b h i d', attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim,):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head,)),
                PreNorm(dim, FeedForward(dim, mlp_dim,))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            st = time.perf_counter()
            x = attn(x) + x
            ed = time.perf_counter()
            # print(f'Attention timing: {ed - st}')

            st = time.perf_counter()
            x = ff(x) + x
            ed = time.perf_counter()
            # print(f'FeedForward timing: {ed - st}')
        return x

