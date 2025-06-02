'''
Date: 2022-03-11 11:01:07
Author: Liu Yahui
LastEditors: Liu Yahui
LastEditTime: 2022-07-13 10:01:25
'''

import math
from einops import rearrange
import torch
import torch.nn as nn


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)



class BHCGSA(nn.Module):

    def __init__(self, dim=16, local_size=2, num_heads=1, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.ls = local_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):

        B, S, D = x.shape
        nl = S // self.ls
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)  # [3, B, h, S, d]

        q_pre = qkv[0].reshape(B * self.num_heads, S, D // self.num_heads).permute(0, 2, 1)  # [B*h, d, S]
        ntimes = int(math.log(nl, 2))
        q_idx_last = torch.arange(S).cuda().unsqueeze(0).expand(B * self.num_heads, S)

        # balanced binary clustering
        for _ in range(ntimes):
            bh, d, n = q_pre.shape  # [B*h*2^n, d, S/2^n]
            q_pre_new = q_pre.reshape(bh, d, 2, n // 2)  # [B*h*2^n, d, 2, S/2^n]
            q_avg = q_pre_new.mean(dim=-1)  # [B*h*2^n, d, 2]
            q_avg = torch.nn.functional.normalize(q_avg.permute(0, 2, 1), dim=-1)
            q_norm = torch.nn.functional.normalize(q_pre.permute(0, 2, 1), dim=-1)

            q_scores = square_distance(q_norm, q_avg)  # [B*h*2^n, S/2^n, 2]
            q_ratio = (q_scores[:, :, 0] + 1) / (q_scores[:, :, 1] + 1)  # [B*h*2^n, S/2^n]
            q_idx = q_ratio.argsort()

            q_idx_last = q_idx_last.gather(dim=-1, index=q_idx).reshape(bh * 2, n // 2)  # [B*h*2^n, S/2^n]
            q_idx_new = q_idx.unsqueeze(1).expand(q_pre.size())  # [B*h*2^n, d, S/2^n]
            q_pre_new = q_pre.gather(dim=-1, index=q_idx_new).reshape(bh, d, 2, n // 2)  # [B*h*2^n, d, 2, S/(2^(n+1))]
            q_pre = rearrange(q_pre_new, 'b d c n -> (b c) d n')  # [B*h*2^(n+1), d, S/(2^(n+1))]

        # clustering is performed independently in each head
        q_idx = q_idx_last.view(B, self.num_heads, S)  # [B, h, S]
        q_idx_rev = q_idx.argsort()  # [B, h, S]

        # cluster query, key, value
        q_idx = q_idx.unsqueeze(0).unsqueeze(4).expand(qkv.size())  # [3, B, h, S, d]
        qkv_pre = qkv.gather(dim=-2, index=q_idx)  # [3, B, h, S, d]
        q, k, v = rearrange(qkv_pre, 'qkv b h (nl ls) d -> qkv (b nl) h ls d', ls=self.ls)
        # MSA
        attn = torch.einsum("blhe,bshe->bhls", q, k)
        attn = self.attn_drop(torch.softmax(self.scale * attn, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", attn, v)

        # merge and reverse
        out = rearrange(out, '(b nl) h ls d -> b h d (nl ls)', h=self.num_heads, b=B)  # [B, h, d, S]
        q_idx_rev = q_idx_rev.unsqueeze(2).expand(out.size())
        res = out.gather(dim=-1, index=q_idx_rev).reshape(B, D, S).permute(0, 2, 1)  # [B, S, D]

        res = self.proj(res)  # [B, S, D]
        res = self.proj_drop(res)

        res = x + res  # [B, S, D]
        return res
