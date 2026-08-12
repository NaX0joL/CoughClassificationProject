from __future__ import annotations

import torch
from torch import nn, Tensor
from typing import Optional

from ...abstract import ModelArchitecture
from .patchtst.backbone import PatchTST_backbone
from .config import PatchTSTConfig
from .series_decomposition import series_decomp



class PatchTST(ModelArchitecture):
    def __init__(self, configs:ArchitectureConfig, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=True, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, 
                 n_normal_heads=0, n_mp_attn_heads=0, qk_weight_share=False, **kwargs):
        
        super().__init__()
        
        self.configs = configs
        default_configs = PatchTSTConfig.default()
        
        c_in = getattr(configs, "enc_in_feature", default_configs.enc_in_feature)
        context_window = getattr(configs, "seq_len", default_configs.seq_len)
        target_window = getattr(configs, "pred_len", default_configs.pred_len)

        n_layers = getattr(configs, "e_layers_num", default_configs.e_layers_num)
        n_heads = getattr(configs, "n_heads_num", default_configs.n_heads_num)
        d_model = getattr(configs, "d_model", default_configs.d_model)
        d_ff = getattr(configs, "d_ff", default_configs.d_ff)
        dropout = getattr(configs, "dropout", default_configs.dropout)
        fc_dropout = getattr(configs, "fc_dropout", default_configs.fc_dropout)
        head_dropout = getattr(configs, "head_dropout", default_configs.head_dropout)
        attn_dropout = getattr(configs, "attn_dropout", default_configs.attn_dropout)

        pre_norm = getattr(configs, "use_pre_norm", default_configs.use_pre_norm)
        attention_output_scaling = getattr(configs, "attention_output_scaling", default_configs.attention_output_scaling)

        individual = getattr(configs, "individual", default_configs.individual)

        patch_len = getattr(configs, "patch_len", default_configs.patch_len)
        stride = getattr(configs, "stride", default_configs.stride)
        padding_patch = getattr(configs, "padding_patch", default_configs.padding_patch)

        revin = getattr(configs, "use_revin", default_configs.use_revin)
        affine = getattr(configs, "use_affine", default_configs.use_affine)
        subtract_last = getattr(configs, "use_subtract_last", default_configs.use_subtract_last)

        use_positional_encoding = getattr(configs, "use_positional_encoding", default_configs.use_positional_encoding)

        decomposition = getattr(configs, "decomposition", default_configs.decomposition)
        kernel_size = getattr(configs, "kernel_size", default_configs.kernel_size)

        head_type = getattr(configs, "head_type", default_configs.head_type)
        bottleneck_dim = getattr(configs, "bottleneck_dim", default_configs.bottleneck_dim)

        self.res_attention = getattr(configs, "res_attention", default_configs.res_attention)
        
        # model definition
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, use_positional_encoding=use_positional_encoding, 
                                  verbose=verbose, 
                                  n_normal_heads=n_normal_heads, n_mp_attn_heads=n_mp_attn_heads, qk_weight_share=qk_weight_share,
                                  bottleneck_dim=bottleneck_dim, attention_output_scaling=attention_output_scaling,
                                  **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, use_positional_encoding=use_positional_encoding, 
                                  verbose=verbose, 
                                  n_normal_heads=n_normal_heads, n_mp_attn_heads=n_mp_attn_heads, qk_weight_share=qk_weight_share,
                                  bottleneck_dim=bottleneck_dim, attention_output_scaling=attention_output_scaling,
                                  **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, use_positional_encoding=use_positional_encoding, 
                                  verbose=verbose, 
                                  n_normal_heads=n_normal_heads, n_mp_attn_heads=n_mp_attn_heads, qk_weight_share=qk_weight_share,
                                  bottleneck_dim=bottleneck_dim, attention_output_scaling=attention_output_scaling,
                                  **kwargs)
        
        self.softmaxed_attn_score = []
        self.attn_score = []
        
        return
    
    def forward(self, x:Tensor) -> Tensor: # x: [Batch, Input length, Channel]
        self.softmaxed_attn_score.clear()
        self.attn_score.clear()

        # quick dimension addition for channel/feature
        x = x.unsqueeze(-1)
        
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        
        # quick dimension removal for channel/feature
        x = x.squeeze(-1)
        
        return x
