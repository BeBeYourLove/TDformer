import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, configs):
        super(MLP, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # Encoder
        self.mlp = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.pred_len)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        out = self.mlp(x_enc.permute(0, 2, 1)).permute(0, 2, 1)

        return out