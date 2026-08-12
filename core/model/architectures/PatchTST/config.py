from dataclasses import dataclass



@dataclass
class PatchTSTConfig:
    seq_len:int
    pred_len:int
    patch_len:int
    stride:int

    e_layers_num:int
    enc_in_feature:int
    d_layers_num:int
    dec_in_feature:int

    n_heads_num:int
    n_normal_heads:int
    n_mp_attn_heads:int
    qk_weight_share:bool
    d_model:int
    d_ff:int

    dropout:float
    fc_dropout:float
    head_dropout:float
    attn_dropout:float

    use_pre_norm:bool

    attention_output_scaling:float=1
    individual:int=0
    padding_patch:str|None=None
    use_revin:bool=False
    use_affine:bool=False
    use_subtract_last:bool=False
    use_positional_encoding:bool=True
    decomposition:int=0
    kernel_size:int=25
    head_type:str="flatten"
    bottleneck_dim:int=128
    res_attention:bool=True

    @classmethod
    def default(cls) -> "PatchTSTConfig":
        patchtst_config = cls(
            seq_len=1000,
            pred_len=1000,
            patch_len=50,
            stride=1,
            
            e_layers_num=1,
            enc_in_feature=1,
            d_layers_num=1,
            dec_in_feature=1,
            
            n_heads_num=1,
            n_normal_heads=0,
            n_mp_attn_heads=0,
            qk_weight_share=False,
            
            d_model=256,
            d_ff=512,
            
            dropout=0.5,
            fc_dropout=0.3,
            head_dropout=0.1,
            attn_dropout=0.1,
            
            use_pre_norm=False,
        )
        return patchtst_config
