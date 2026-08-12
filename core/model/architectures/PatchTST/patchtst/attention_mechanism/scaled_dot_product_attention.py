import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
from typing import Optional



class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa
        self.attention_maps:Optional[Tensor] = None

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q                   : [bs, n_heads, max_q_len. d_k]
            k                   : [bs, n_heads, d_k, seq_len]
            v                   : [bs, n_heads, seq_len, d_v]
            prev                : [bs, n_heads, q_len, seq_len]
            key_padding_mask    : [bs, seq_len]
            attn_mask           : [1, seq_len, seq_len]
        Output shape:
            output              : [bs, n_heads, q_len, d_v]
            attn                : [bs, n_heads, q_len, seq_len]
            scores              : [bs, n_heads, q_len, seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores: [bs, n_heads, max_q_len, seq_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:       # attn_mask with shape [q_len, seq_len]
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:        # mask with shape [bs, q_len]
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)
        
        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)       # attn_weights   : [bs, n_heads, max_q_len, seq_len]
        attn_weights = self.attn_dropout(attn_weights)
        self.attention_maps = attn_weights
        
        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)      # output: [bs, n_heads, max_q_len, d_v]                               

        if self.res_attention: 
            return output, attn_weights, attn_scores
        else: 
            return output, attn_weights



if __name__ == "__main__":
    print("Done!")
