import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from math import sqrt
import os

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

"""
Full Attention
"""
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=3, scale=None, attention_dropout=0.1, T=1, activation='softmax',
                 output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.activation = activation
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.T = T

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * scale

        if self.activation == 'softmax':
            if self.mask_flag:
                if attn_mask is None:
                    attn_mask = TriangularCausalMask(B, L, device=queries.device)

                scores.masked_fill_(attn_mask.mask, -np.inf)

            A = self.dropout(torch.softmax(scores / self.T, dim=-1))
            V = torch.einsum("bhls,bshd->blhd", A, values)

        elif self.activation == 'linear':
            V = torch.einsum("bhls,bshd->blhd", scores, values)

        elif self.activation == 'linear_norm':
            mins = scores.min(dim=-1)[0].unsqueeze(-1).expand(-1, -1, -1, scores.shape[3])
            scores = scores - mins + 1e-8

            if self.mask_flag:
                if attn_mask is None:
                    attn_mask = TriangularCausalMask(B, L, device=queries.device)
                scores.masked_fill_(attn_mask.mask, 0)

            sums = scores.sum(dim=-1).unsqueeze(-1).expand(-1, -1, -1, scores.shape[3])
            scores /= sums
            V = torch.einsum("bhls,bshd->blhd", scores, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

"""
Fourier Attention
"""
class FourierAttention(nn.Module):
    def __init__(self, T=1, activation='softmax', output_attention=False):
        super(FourierAttention, self).__init__()
        print(' fourier enhanced cross attention used!')
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.    
        """
        self.activation = activation
        self.output_attention = output_attention
        self.T = T

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        _, S, H, E = k.shape
        xq = q.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)

        xq_ft_ = torch.fft.rfft(xq, dim=-1, norm='ortho')
        xk_ft_ = torch.fft.rfft(xk, dim=-1, norm='ortho')
        xv_ft_ = torch.fft.rfft(xv, dim=-1, norm='ortho')

        xqk_ft = torch.einsum("bhex,bhey->bhxy", xq_ft_, torch.conj(xk_ft_)) / sqrt(E)

        if self.activation == 'softmax':
            xqk_ft = torch.softmax(xqk_ft.abs() / self.T, dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
            xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xv_ft_)

        elif self.activation == 'linear':
            xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xv_ft_)

        elif self.activation == 'linear_norm':
            mins_real = xqk_ft.real.min(dim=-1)[0].unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft_real = xqk_ft.real - mins_real
            sums_real = xqk_ft_real.sum(dim=-1).unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft_real /= sums_real

            mins_imag = xqk_ft.imag.min(dim=-1)[0].unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft_imag = xqk_ft.imag - mins_imag
            sums_imag = xqk_ft_imag.sum(dim=-1).unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft_imag /= sums_imag

            xqkv_ft_real = torch.einsum("bhxy,bhey->bhex", xqk_ft_real, xv_ft_.real)
            xqkv_ft_imag = torch.einsum("bhxy,bhey->bhex", xqk_ft_imag, xv_ft_.imag)
            xqkv_ft = torch.complex(xqkv_ft_real, xqkv_ft_imag)

        elif self.activation == 'linear_norm_abs':
            xqk_ft = xqk_ft.abs() / xqk_ft.abs().sum(dim=-1).unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
            xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xv_ft_)

        elif self.activation == 'linear_norm_real':
            mins_real = xqk_ft.real.min(dim=-1)[0].unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft_real = xqk_ft.real - mins_real
            sums_real = xqk_ft_real.sum(dim=-1).unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft_real /= sums_real

            xqk_ft = torch.complex(xqk_ft_real, torch.zeros_like(xqk_ft_real))
            xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xv_ft_)

        out = torch.fft.irfft(xqkv_ft, n=L, dim=-1, norm='ortho').permute(0, 3, 1, 2)

        if self.output_attention == False:
            return (out, None)
        else:
            return (out, (xqk_ft_real, xqk_ft_imag))


