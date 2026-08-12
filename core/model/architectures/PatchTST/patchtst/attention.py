import torch
from torch import nn, Tensor
from typing import Optional

from .attention_mechanism.scaled_dot_product_attention import _ScaledDotProductAttention
from .attention_mechanism.matrix_profile_attention import _MatrixProfileAttention



class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False,
                 n_normal_heads=0, n_mp_attn_heads=0, qk_weight_share=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        
        # different head type num control
        if n_normal_heads + n_mp_attn_heads != n_heads:
            if n_normal_heads + n_mp_attn_heads < n_heads:
                n_normal_heads += n_heads - n_normal_heads - n_mp_attn_heads
            elif n_normal_heads + n_mp_attn_heads > n_heads:
                n_normal_heads = n_heads // 2
                n_mp_attn_heads = n_heads - n_normal_heads
        self.n_normal_heads = n_normal_heads
        self.n_mp_attn_heads = n_mp_attn_heads

        # the three infamous vector k, q, v
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        if qk_weight_share:
            self.W_K = self.W_Q                                                 # designate as a copy
        else:
            self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)         # create a separate individual layer
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads) 
        self.res_attention = res_attention
        if n_normal_heads > 0:
            self.sdp_attn = _ScaledDotProductAttention(d_model, n_normal_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)
        if n_mp_attn_heads > 0:
            self.mp_attn = _MatrixProfileAttention(d_model, n_mp_attn_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa, 
                                                    qk_weight_share=qk_weight_share)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs, n_heads, max_q_len, d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs, n_heads, d_k, q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs, n_heads, q_len, d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        # output    : [bs, n_heads, q_len, d_v]
        # attn      : [bs, n_heads, q_len, q_len]
        # scores    : [bs, n_heads, max_q_len, q_len]
        
        # slice q, k, v for mixed attn heads
        slice_index = self.n_normal_heads
        q_sdp, q_mp = q_s[:, :slice_index], q_s[:, slice_index:]
        k_sdp, k_mp = k_s[:, :slice_index], k_s[:, slice_index:]
        v_sdp, v_mp = v_s[:, :slice_index], v_s[:, slice_index:]
        
        prev_sdp = prev[:, :slice_index] if prev is not None else None
        prev_mp  = prev[:, slice_index:] if prev is not None else None
        
        if self.n_normal_heads > 0:
            if self.res_attention:
                sdp_attn_output, sdp_attn_weights, sdp_attn_scores = self.sdp_attn(q_sdp, k_sdp, v_sdp, prev=prev_sdp, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            else:
                sdp_attn_output, sdp_attn_weights = self.sdp_attn(q_sdp, k_sdp, v_sdp, prev=prev_sdp, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.n_mp_attn_heads > 0:
            if self.res_attention:
                mp_attn_output, mp_attn_weights, mp_attn_scores = self.mp_attn(q_mp, k_mp, v_mp, prev=prev_mp, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            else:
                mp_attn_output, mp_attn_weights = self.mp_attn(q_mp, k_mp, v_mp, prev=prev_mp, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        # merge output back to one
        if self.n_normal_heads > 0 and self.n_mp_attn_heads > 0:
            output = torch.cat([sdp_attn_output, mp_attn_output], dim=1)
        elif self.n_normal_heads > 0:
            output = sdp_attn_output
        elif self.n_mp_attn_heads > 0:
            output = mp_attn_output
            
        if self.n_normal_heads > 0 and self.n_mp_attn_heads > 0:
            #attn_weights = torch.cat([sdp_attn_weights, mp_attn_weights], dim=1)
            attn_weights = None
        elif self.n_normal_heads > 0:
            attn_weights = sdp_attn_weights
        elif self.n_mp_attn_heads > 0:
            attn_weights = mp_attn_weights
            
        if self.res_attention:
            if self.n_normal_heads > 0 and self.n_mp_attn_heads > 0:
                attn_scores = torch.cat([sdp_attn_scores, mp_attn_scores], dim=1)
            elif self.n_normal_heads > 0:
                attn_scores = sdp_attn_scores
            elif self.n_mp_attn_heads > 0:
                attn_scores = mp_attn_scores
        
        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)      # output: [bs, q_len, n_heads*d_v]
        output = self.to_out(output)
        
        if self.res_attention: 
            return output, attn_weights, attn_scores
        else: 
            return output, attn_weights



if __name__ == "__main__":
    print("Done!")