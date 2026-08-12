import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional



class _MatrixProfileAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False, qk_weight_share=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa
        self.qk_weight_share = qk_weight_share
    
    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q                   : [bs, n_heads, seq_len, d_k]
            k                   : [bs, n_heads, d_k, seq_len]
            v                   : [bs, n_heads, seq_len, d_v]
            prev                : [bs, n_heads, q_len, seq_len]
            key_padding_mask    : [bs, seq_len]
            attn_mask           : [1, seq_len, seq_len]
        Output shape:
            output              : [bs, n_heads, seq_len, d_v]
            attn                : [bs, n_heads, seq_len, seq_len]
            scores              : [bs, n_heads, seq_len, seq_len]
        '''
        
        assert q.shape[-2]==k.shape[-1] and k.shape[-1]==q.shape[-2], "k and q shape mismatch"

        # extract the dim name
        bs, n_heads, seq_len, d_k = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
        
        # combine the dim that are not cdist-ed
        q_reshaped = q.reshape(bs*n_heads, seq_len, d_k)                        # q: [bs*n_heads, d_k, seq_len]
        k_reshaped = k.permute(0, 1, 3, 2).reshape(bs*n_heads, seq_len, d_k)    # k: [bs*n_heads, d_k, seq_len]
        
        # compute cdist
        dm = torch.cdist(q_reshaped, k_reshaped, p=2) * self.scale              # dm: [bs*n_heads, seq_len, seq_len]
        
        # split back the dim that are not cdist-ed
        dm = dm.reshape(bs, n_heads, seq_len, seq_len)                          # dm: [bs, n_heads, seq_len, seq_len]
        
        # mask diagonal element if using weight share
        if self.qk_weight_share:
            # create mask to remove diag element
            diag_mask = torch.eye(dm.shape[2], dtype=torch.bool)                # diag_mask: [seq_len x seq_len]
            diag_mask = diag_mask.unsqueeze(0).unsqueeze(0)                     # diag_mask: [1 x 1 x seq_len x seq_len]
            diag_mask = diag_mask.to(dm.device)
            
            # mask the diag element
            dm = dm.masked_fill(diag_mask, float('inf'))
            
        else:
            # do nothing, but have the diag_mask defined
            diag_mask = torch.zeros(dm.shape[2], dm.shape[2], dtype=torch.bool)
            diag_mask = diag_mask.unsqueeze(0).unsqueeze(0)                     # diag_mask: [1 x 1 x seq_len x seq_len]
            diag_mask = diag_mask.to(dm.device)
        
        # take min value in each row
        min_row, _ = torch.min(dm, dim=-1)                                      # min_row: [bs, n_heads, seq_len]
        dm = dm.masked_fill(diag_mask, float(0))        # masked it to 0 for loss comparison
        
        # get the probabilistic value
        #modified_min_row = F.softmax(min_row, dim=-1)                          # softmaxed_min_row: [bs, n_heads, seq_len]
        
        #####@@@@@ tmp mod for test later
        if True:
            modified_min_row = F.softmax(min_row, dim=-1)
        if False:
            modified_min_row = torch.sigmoid(min_row)
        if False:
            # need to think if its appropriate to scale globally
            # or choose to scale per batch or per head
            def min_max_norm(input:Tensor):
                tensor_min = input.min(dim=-1, keepdim=True).values
                tensor_max = input.max(dim=-1, keepdim=True).values
                eps = 1e-6
                output = (input - tensor_min) / (tensor_max - tensor_min + eps)
                return output
            modified_min_row = min_max_norm(min_row)
        if False:
            # also need to think which dim to do mean and std on
            def z_norm(input:Tensor):
                mean = input.mean(dim=-1, keepdim=True)
                std = input.std(dim=-1, keepdim=True)
                eps = 1e-6
                output = (input - mean) / (std + eps)
                return output
            modified_min_row = z_norm(min_row)
        if False:
            modified_min_row = min_row
        
        # mult back with the v vector
        modified_min_row = modified_min_row.unsqueeze(-1)
        output = torch.mul(-modified_min_row, v)                                # output: [bs, n_heads, seq_len, d_v]
        
        # redeclaration for clarity
        output = output                                                         # output: [bs, n_heads, seq_len, d_v]
        attn_weights = modified_min_row                                         # attn_weights: [bs, n_heads, seq_len, seq_len]
        attn_scores = dm                                                        # attn_scores: [bs, n_heads, seq_len, seq_len]
        
        if self.res_attention: 
            return output, attn_weights, attn_scores
        else: 
            return output, attn_weights



if __name__ == "__main__":
    print("Done!")