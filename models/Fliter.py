import torch
from torch import nn
from layers.GCN_blocks import mlp

class Filter(nn.Module):
    def __init__(self, args, out_channals):
        super().__init__()

        self.embed_size = args.seq_len
        self.hidden_dim = self.embed_size * 4
        self.scale = 0.2
        self.w0 = nn.Parameter(self.scale * torch.randn([self.embed_size]))
        self.mlp = mlp(c_in=self.embed_size, c_out=out_channals, c_hid=self.hidden_dim)
    
    def circular_convolution(self, x):  # x: [B, N, T]
        x = torch.fft.rfft(x, dim=-1, norm='ortho')
        w0 = torch.fft.rfft(self.w0, dim=-1, norm='ortho')
        y = x * w0
        out = torch.fft.irfft(y, self.embed_size, dim=2, norm='ortho')
        return out

        # xflip = torch.flip(x, dims=[-1])
        # x = torch.cat([xflip, x, xflip], dim=-1)
        # nx, nw = x.size(-1), self.w0.size(-1)
        # n = nx + nw -1
        # x = torch.fft.rfft(x, n=n, dim=-1, norm='ortho')
        # w0 = torch.fft.rfft(self.w0, n=n, dim=-1)
        # y = x * w0
        # out = torch.fft.irfft(y, n=n, dim=2, norm='ortho')[..., 18:30]
        # return out
    
    def forward(self, x):
        
        x_filter = x#self.circular_convolution(x)
        x_res =self.mlp(x_filter)
        return x_res, x_filter