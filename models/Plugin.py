import torch
from torch import nn
from layers.GCN_blocks import my_attention


class Plugin(nn.Module):
    def __init__(self, args, channel):
        super(Plugin, self).__init__()
        self.args = args

        self.hist_len = args.seq_len
        self.pred_len = args.pred_len

        self.time_dim = 5

        self.key_dim = 128
        self.val_n = 128 * 4 * 4
        self.val_dim = 128
        self.time_val = nn.Parameter(torch.randn([self.val_n, self.val_dim]))

        self.ln_t = nn.Linear(self.time_dim, args.node_num)
        self.norm = torch.tanh
        self.ln_q_map = nn.Linear(self.pred_len, self.key_dim)
        self.att_w = my_attention(node_q=channel, node_v=self.val_n,
            input_dim_q=self.key_dim, input_dim_k=self.val_dim, input_dim_v=self.val_dim,
            output_dim=self.key_dim)
        self.ln_ps = nn.Linear(self.pred_len, self.key_dim)
        self.ln_p = nn.Linear(self.val_dim, self.hist_len)

    def forward(self, x_enc_true, x_mark_enc, x_dec_pred, x_mark_dec):
        means = torch.mean(x_enc_true, dim=1, keepdim=True)
        stdev = torch.std(x_enc_true, dim=1, keepdim=True) + 1e-3
        x_enc_true = (x_enc_true - means) / stdev
        x_dec_pred = (x_dec_pred - means) / stdev

        # map 参数共享
        x_enc_map = self.ln_t(x_mark_enc[..., :self.time_dim])
        x_dec_map = self.ln_t(x_mark_dec[..., :self.time_dim])
        x_enc_map = self.norm(x_enc_map)
        x_dec_map = self.norm(x_dec_map)

        # combine
        q_enc_map = self.ln_q_map(x_enc_map.permute(0,2,1)) # b, n, dim
        q_dec_map = self.ln_q_map(x_dec_map.permute(0,2,1))
        w_enc = self.att_w(q_enc_map, self.time_val, self.time_val) # [b,n,dim] [n, dim] -> [b,n,dim]
        w_dec = self.att_w(q_dec_map, self.time_val, self.time_val)
        p_enc = self.ln_ps(x_enc_true.permute(0,2,1))
        p_dec = self.ln_ps(x_dec_pred.permute(0,2,1))
        e_enc = p_enc - w_enc
        e_dec = p_dec - w_dec
        e_enc_means = torch.mean(e_enc, dim=2, keepdim=True)
        e_enc_std = torch.std(e_enc, dim=2, keepdim=True) + 1e-6
        e_dec_means = torch.mean(e_dec, dim=2, keepdim=True)
        e_dec_std = torch.std(e_dec, dim=2, keepdim=True) + 1e-6
        e_dec = (e_dec - e_dec_means) / e_dec_std * e_enc_std + e_enc_means 
        pred = self.ln_p(e_dec + w_dec).permute(0,2,1)

        pred = pred * stdev + means

        return pred