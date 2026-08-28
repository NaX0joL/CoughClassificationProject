from typing import Optional

from torch import Tensor

from ...abstract import ModelArchitecture
from .patchtst.backbone import PatchTST_backbone
from .series_decomposition import series_decomp


class PatchTST(ModelArchitecture):

    def __init__(
        self,
        seq_len:int,
        pred_len:int,
        patch_len:int,
        stride:int,
        enc_in_feature:int,
        e_layers_num:int,
        n_heads_num:int,
        d_model:int,
        d_ff:int,
        dropout:float,
        fc_dropout:float,
        head_dropout:float,
        attn_dropout:float,
        use_pre_norm:bool,
        max_seq_len:Optional[int]=1024,
        d_k:Optional[int]=None,
        d_v:Optional[int]=None,
        norm:str="BatchNorm",
        act:str="gelu",
        key_padding_mask:bool|str="auto",
        padding_var:Optional[int]=None,
        attention_mask:Optional[Tensor]=None,
        res_attention:bool=True,
        store_attn:bool=True,
        pe:str="zeros",
        learn_pe:bool=True,
        pretrain_head:bool=False,
        head_type:str="flatten",
        individual:bool=False,
        padding_patch:str|None=None,
        use_revin:bool=False,
        use_affine:bool=False,
        use_subtract_last:bool=False,
        use_positional_encoding:bool=True,
        decomposition:bool=False,
        kernel_size:int=25,
        bottleneck_dim:int=128,
        n_normal_heads:int=0,
        n_mp_attn_heads:int=0,
        qk_weight_share:bool=False,
        attention_output_scaling:float=1,
        verbose:bool=False,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        
        self.enc_in_feature = enc_in_feature
        self.e_layers_num = e_layers_num
        self.n_heads = n_heads_num
        
        self.n_normal_heads = n_normal_heads
        self.n_mp_attn_heads = n_mp_attn_heads
        self.qk_weight_share = qk_weight_share
        self.attention_output_scaling = attention_output_scaling
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.fc_dropout = fc_dropout
        self.head_dropout = head_dropout
        
        self.decomposition = decomposition
        self.res_attention = res_attention
        self.use_positional_encoding = use_positional_encoding

        backbone_parameters = {
            "c_in": enc_in_feature,
            "context_window": seq_len,
            "target_window": pred_len,
            "patch_len": patch_len,
            "stride": stride,
            "max_seq_len": max_seq_len,
            "n_layers": e_layers_num,
            "d_model": d_model,
            "n_heads": n_heads_num,
            "d_k": d_k,
            "d_v": d_v,
            "d_ff": d_ff,
            "norm": norm,
            "attn_dropout": attn_dropout,
            "dropout": dropout,
            "act": act,
            "key_padding_mask": key_padding_mask,
            "padding_var": padding_var,
            "attn_mask": attention_mask,
            "res_attention": res_attention,
            "pre_norm": use_pre_norm,
            "store_attn": store_attn,
            "pe": pe,
            "learn_pe": learn_pe,
            "fc_dropout": fc_dropout,
            "head_dropout": head_dropout,
            "padding_patch": padding_patch,
            "pretrain_head": pretrain_head,
            "head_type": head_type,
            "individual": individual,
            "revin": use_revin,
            "affine": use_affine,
            "subtract_last": use_subtract_last,
            "use_positional_encoding": use_positional_encoding,
            "verbose": verbose,
            "n_normal_heads": n_normal_heads,
            "n_mp_attn_heads": n_mp_attn_heads,
            "qk_weight_share": qk_weight_share,
            "bottleneck_dim": bottleneck_dim,
            "attention_output_scaling": attention_output_scaling,
        }

        if decomposition:
            self.decomposition_layer = series_decomp(kernel_size)
            self.trend_model = PatchTST_backbone(**backbone_parameters)
            self.residual_model = PatchTST_backbone(**backbone_parameters)
        else:
            self.model = PatchTST_backbone(**backbone_parameters)

        self.softmaxed_attn_score:list[Tensor] = []
        self.attn_score:list[Tensor] = []
        return

    def forward(self, x:Tensor) -> Tensor:
        self.softmaxed_attn_score.clear()
        self.attn_score.clear()

        if x.ndim == 2:
            if self.enc_in_feature != 1:
                raise ValueError(
                    "PatchTST requires [batch, sequence, features] when "
                    "enc_in_feature is not 1",
                )
            x = x.unsqueeze(1)
        elif x.ndim == 3:
            if x.shape[2] != self.enc_in_feature:
                raise ValueError(
                    "PatchTST input feature count does not match enc_in_feature",
                )
            x = x.permute(0, 2, 1)
        else:
            raise ValueError(
                "PatchTST requires values with shape [batch, sequence, features]",
            )

        if x.shape[2] != self.seq_len:
            raise ValueError("PatchTST input sequence length does not match seq_len")

        if self.decomposition:
            residual, trend = self.decomposition_layer(x)
            x = self.residual_model(residual) + self.trend_model(trend)
        else:
            x = self.model(x)

        self._store_attention_scores()
        return x.mean(dim=1)

    def _store_attention_scores(self) -> None:
        source_model = self.residual_model if self.decomposition else self.model
        self.softmaxed_attn_score = source_model.softmaxed_attn_score
        if self.res_attention:
            self.attn_score = source_model.attn_score
        return
