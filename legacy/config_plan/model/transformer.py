from core.model import ModelConfig
from core.model.architectures.PatchTST import PatchTST
from core.model.behavior.classification_behavior import ClassificationBehavior



transformer_config = ModelConfig(
    architecture=PatchTST(
        seq_len=42,
        pred_len=2,
        patch_len=1,
        stride=1,
        
        enc_in_feature=40,
        e_layers_num=3,
        n_heads_num=4,
        
        d_model=128,
        d_ff=256,
        
        dropout=0.3,
        fc_dropout=0.2,
        head_dropout=0.2,
        attn_dropout=0.1,
        
        use_pre_norm=False,
    ),
    behavior=ClassificationBehavior(),
)
