import torch
from torch import nn, Tensor
from typing import Optional

from .attention import _MultiheadAttention
from .components.transpose import Transpose
from .components.additional_functions import get_activation_fn, positional_encoding



class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False,
                 n_normal_heads=0, n_mp_attn_heads=0, qk_weight_share=False, attention_output_scaling=1):
        
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention, 
                                             n_normal_heads=n_normal_heads, n_mp_attn_heads=n_mp_attn_heads, qk_weight_share=qk_weight_share)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.attention_output_scaling = attention_output_scaling


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:
        # multi head attention pre-norm
        if self.pre_norm:
            src = self.norm_attn(src)
            
        # print("after prenorm std:", src.std().item())
            
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            # manually save the attention score
            self.attn = attn
            self.scores = scores
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            # manually save the attention score
            self.attn = attn
        
        # print("after multihead attn std:", src2.std().item())
        
        ##########@@@#DEBUG - print tensor norm from attn/skip conn.
        if False:
            print("skip conn. max  : ", src.max())
            print("skip conn. norm : ", src.norm())
            print("attn max        : ", src2.max())
            print("attn norm       : ", src2.norm())
        
        # attn outpute scaling
        scale = self.attention_output_scaling
        src2 = src2 * scale
        
        ## Add & Norm - attn skip connection
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        # multi head attention post-norm
        if not self.pre_norm:
            src = self.norm_attn(src)

        # print("after skip conn std:", src.std().item())
        
        # ffn pre-norm
        if self.pre_norm:
            src = self.norm_ffn(src)
            
        # print("after ffn prenorm std:", src.std().item())
        
        ## Position-wise Feed-Forward Network
        src2 = self.ff(src)
        
        # print("after ffn std:", src2.std().item())
        
        ## Add & Norm - feed forward skip connection
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        # ffn post-norm
        if not self.pre_norm:
            src = self.norm_ffn(src)

        # print("after skip conn2 std:", src.std().item())
        
        if self.res_attention:
            return src, scores
        else:
            return src



class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=True, n_layers=1, pre_norm=False,
                 n_normal_heads=0, n_mp_attn_heads=0, qk_weight_share=False,
                 attention_output_scaling=1):
        
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm,
                                                      n_normal_heads=n_normal_heads, 
                                                      n_mp_attn_heads=n_mp_attn_heads,
                                                      qk_weight_share=qk_weight_share,
                                                      attention_output_scaling=attention_output_scaling) for i in range(n_layers)])
        
        self.res_attention = res_attention
        
        self.softmaxed_attn_score = []
        self.attn_score = []

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        # restart attention for every forward pass
        self.softmaxed_attn_score.clear()
        self.attn_score.clear()
        
        output = src
        scores = None
        # if self.res_attention:
        #     for mod in self.layers: 
        #         output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # else:
        #     for mod in self.layers: 
        #         output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                
        for mod in self.layers:
            if self.res_attention:
                output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                # manually store attention matrix value after each forward pass 
                # self.softmaxed_attn_score.append(mod.attn.detach())
                # self.attn_score.append(mod.scores.detach())
                self.softmaxed_attn_score.append(None)
                self.attn_score.append(None)
            else:
                output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                # manually store attention matrix value after each forward pass 
                # self.softmaxed_attn_score.append(mod.attn.detach())
                self.softmaxed_attn_score.append(None)
        
        return output



class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu",
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, 
                 use_positional_encoding=True,
                 n_normal_heads=0, n_mp_attn_heads=0, qk_weight_share=False, attention_output_scaling=1,
                 **kwargs):
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        self.res_attention = res_attention
        
        self.use_positional_encoding = use_positional_encoding
        
        # Input encoding // mapping of patches into embedding vector
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers,
                                   n_normal_heads=n_normal_heads, n_mp_attn_heads=n_mp_attn_heads, qk_weight_share=qk_weight_share, 
                                   attention_output_scaling=attention_output_scaling)
        
        self.softmaxed_attn_score = []
        self.attn_score = []
        

    def forward(self, x) -> Tensor:                                             # x: [bs x nvars x patch_len x patch_num]
        self.softmaxed_attn_score.clear()
        self.attn_score.clear()
        
        # print("tstiencoder input std:", x.std().item())
        
        n_vars = x.shape[1]
        # dim adjustment for encoder
        x = x.permute(0,1,3,2)                                                  # x: [bs x nvars x patch_num x patch_len]
        
        # token to embedding transformation
        x = self.W_P(x)                                                         # x: [bs x nvars x patch_num x d_model]
        
        # print("after W_P std:", x.std().item())
        
        #########@@@ DEBUG: get embedding tensor before everything else
        if False :
            patch_embeddings = x.clone()
            patch_embeddings = None

        # combine the first two dim for encoder input
        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))     # u: [bs * nvars x patch_num x d_model]
        
        # positional encoding
        if self.use_positional_encoding:
            u = u + self.W_pos
        
        # print("after pos encoding std:", u.std().item())
        
        u = self.dropout(u)                                                     # u: [bs * nvars x patch_num x d_model]

        # print("after dropout std:", u.std().item())

        # Encoder
        z = self.encoder(u)                                                     # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))               # z: [bs x nvars x patch_num x d_model]
        
        # print("after encoder std:", z.std().item())
        
        #########@@@ DEBUG: get embedding tensor before everything else
        if False:
            final_embeddings = z.clone()
            final_embeddings = None
        
        # dim adjustment for transformer head
        z = z.permute(0,1,3,2)                                                  # z: [bs x nvars x d_model x patch_num]
        
        # manually save attention matrix
        self.softmaxed_attn_score = self.encoder.softmaxed_attn_score
        if self.res_attention:
            self.attn_score = self.encoder.attn_score
        
        # DEBUG: save embedding as class attributes
        if False:
            self.patch_embeddings = patch_embeddings
            self.final_embeddings = final_embeddings
        
        # print("tstiencoder output std:", z.std().item())
        
        return z
