import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = nn.Linear(c_in, c_out, bias=True)
    
    def forward(self, x):
        return self.mlp(x)

class mlp(nn.Module):
    def __init__(self, c_in, c_out, c_hid):
        super().__init__()
        self.l1 = linear(c_in, c_hid)
        self.l2 = linear(c_hid, c_out)
        nn.init.kaiming_normal_(self.l1.mlp.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.l2.mlp.weight, nonlinearity='relu')
        
    def forward(self, x):
        y = self.l1(x)
        y = torch.relu(y)
        y = self.l2(y)
        return y

class AVWGCN(nn.Module):
    def __init__(self, in_channels, out_channels, node_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cheb_k = 3

        self.weights_pool = nn.Parameter(torch.rand(node_dim, self.cheb_k, in_channels, out_channels))
        self.bias_pool = nn.Parameter(torch.zeros(node_dim, out_channels))
    
    def forward(self, x, node_embeddings):   
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)  
        support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):  
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # b, N, dim_out
        return x_gconv

class my_gcn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(my_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # learnable parameters
        self.weight = nn.Parameter(torch.rand(self.in_channels, self.out_channels))
        self.bias = nn.Parameter(torch.zeros(self.out_channels))

    def forward(self, x, adj_matrix):
        """
        :param x: 输入特征张量，形状为[B, N, H]，其中B是batch_size，N是节点数量，H是输入特征维度
        :param adj_matrix: 邻接矩阵，形状为[N, N]，表示图的连接关系
        :return: 输出特征张量，形状为[B, N, H]
        """

        # 将邻接矩阵扩展为适合批量处理的形状 [B, N, N]
        batch_size = x.size(0)
        adj_matrix = adj_matrix.unsqueeze(0).repeat(batch_size, 1, 1)

        # 进行图卷积操作
        h = torch.matmul(x, self.weight)  # [B, N, out_channels]
        output = torch.matmul(adj_matrix, h) + self.bias  # [B, N, out_channels]

        return output

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # learnable parameters
        self.weight = nn.Parameter(torch.rand(self.in_channels, self.out_channels))
        self.bias = nn.Parameter(torch.zeros(self.out_channels))

    def forward(self, x, adj_matrix):
        """
        :param x: 输入特征张量，形状为[B, N, H]，其中B是batch_size，N是节点数量，H是输入特征维度
        :param adj_matrix: 邻接矩阵，形状为[N, N]，表示图的连接关系
        :return: 输出特征张量，形状为[B, N, H]
        """

        # 将邻接矩阵扩展为适合批量处理的形状 [B, N, N]
        batch_size = x.size(0)
        adj_matrix = adj_matrix.unsqueeze(0).repeat(batch_size, 1, 1)

        # 进行图卷积操作
        h = torch.matmul(x, self.weight)  # [B, N, out_channels]
        output = torch.matmul(adj_matrix, h) + self.bias  # [B, N, out_channels]

        return output

class my_attention(nn.Module):
    """
    input:
        x -> query [batch, node_q, input_dim_q],
        x -> key   [batch, node_v, input_dim_k],
        x -> value [batch, node_v, input_dim_v],
    
    process:
        query = fc(query) -> [batch, node_q, att_dim],
        key = fc(key)     -> [batch, node_k, att_dim],
        a = query@key   -> [batch, node_q, node_v],
        value = fc(value) -> [batch, node_v, att_dim]
        out = a@value     -> [batch, node_q, att_dim]
        out = out_proj(out)    -> [batch, node_q, output_dim]
    """
    def __init__(self, node_q, node_v, input_dim_q, input_dim_k, input_dim_v, output_dim, mask=False, n_head=1,
                FC_Q=None, FC_K=None, FC_V=None):
        super().__init__()
        self.head = n_head
        self.input_dim_q = input_dim_q
        self.input_dim_k = input_dim_k
        self.input_dim_v = input_dim_v
        self.node_q = node_q
        self.node_v = node_v
        self.mask = mask
        self.att_num = 32

        self.FC_Q = linear(input_dim_q, self.att_num * self.head) if FC_Q is None else FC_Q
        self.FC_K = linear(input_dim_k, self.att_num * self.head) if FC_K is None else FC_K
        self.FC_V = linear(input_dim_v, self.att_num * self.head) if FC_V is None else FC_V
        self.out_proj = linear(self.head*self.att_num, output_dim)
    
    def forward(self, query, key, value):
        b = query.shape[0]
        query = self.FC_Q(query)    # B, M, D*n_h
        key = self.FC_K(key)        # B, N, D*n_h
        value = self.FC_V(value)    # B, N, D*n_h

        if len(query.shape)==2: query = query.unsqueeze(dim=0).repeat(b,1,1)
        query = query.view(b, -1, self.head, self.att_num).permute(2,0,1,3).contiguous().view(b*self.head, -1, self.att_num)  # B*n_h, M, D
        if len(key.shape)==2: key = key.unsqueeze(dim=0).repeat(b,1,1)
        key = key.view(b, -1, self.head, self.att_num).permute(2,0,1,3).contiguous().view(b*self.head, -1, self.att_num)
        if len(value.shape)==2: value = value.unsqueeze(dim=0).repeat(b,1,1)
        value = value.view(b, -1, self.head, self.att_num).permute(2,0,1,3).contiguous().view(b*self.head, -1, self.att_num)

        att_score = torch.einsum('bmd,bnd->bmn', [query, key])  # B*n_h, M, N

        if self.mask:
            num = att_score.shape[-1]
            mask = torch.ones( num, num, dtype=torch.bool, device=query.device ).tril()
            att_score.masked_fill_(~mask, -torch.inf)
        
        att_score = att_score/math.sqrt(self.att_num)
        
        att_score = torch.softmax(att_score, dim=-1) # [att_dim, att_dim]

        # print('att_score:', att_score.shape)
        # print('value:', value.shape)
        out = torch.einsum('bmn,bnd->bmd', [att_score, value])
        out = out.view(self.head, b, -1, self.att_num).permute(1,2,0,3).contiguous().view(b, -1, self.head*self.att_num)
        out = self.out_proj(out)
        # out = self.LayerNorm(out)
        return out

######################
#   time2vec
######################

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], -1)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    def __init__(self, activation, hiddem_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, hiddem_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, hiddem_dim)

        self.fc1 = nn.Linear(hiddem_dim, 2)

    def forward(self, x):
        x = self.l1(x)
        x = self.fc1(x)
        return x
