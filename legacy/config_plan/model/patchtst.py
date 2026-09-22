from core.model import ModelConfig
from core.model.architectures.PatchTST import PatchTST
from core.model.behavior.classification_behavior import ClassificationBehavior



patchtst_config = ModelConfig(
    architecture=PatchTST(
        seq_len=820,
        pred_len=2,
        patch_len=50,
        stride=10,
        
        enc_in_feature=20,
        e_layers_num=2,
        n_heads_num=4,
        
        d_model=256,
        d_ff=512,
        
        dropout=0.5,
        fc_dropout=0.3,
        head_dropout=0.1,
        attn_dropout=0.1,
        
        use_pre_norm=False,
    ),
    behavior=ClassificationBehavior(),
)
