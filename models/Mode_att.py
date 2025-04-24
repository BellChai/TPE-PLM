import os
import torch
from torch import nn
import numpy as np
from sklearn.cluster import KMeans
from layers.GCN_blocks import mlp

class Mode_att(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.Q_bias = args.Q_bias
        self.week_opt = args.week_opt

        self.seq_len = args.seq_len
        self.node_num = args.node_num
        self.t_n = args.cluster
        self.k, self.v, self.kw, self.vw = self.get_kv(args.train_data)  # N, vn, tn, T
        self.k, self.v = self.k.to(args.device), self.v.to(args.device)
        if self.week_opt:
            self.kw, self.vw = self.kw.to(args.device), self.vw.to(args.device)
        # self.Q_bias = nn.Parameter( torch.randn([1]) )

        self.att_weight = nn.Parameter( torch.randn([self.node_num, self.t_n+1+ 2*args.week_opt ]))
        self.att_bias = nn.Parameter( torch.randn([self.node_num]) / 5)

    
    def get_kv(self, data): # data [L, N] np
        if self.week_opt:
            cluster_path = f'dataset/PEMS/{self.args.model_comment}_cluster={self.t_n}.pt'
            cluster_week_path = f'dataset/PEMS/{self.args.model_comment}_cluster=2_week.pt'
            if os.path.exists(cluster_path):
                centers = torch.load(cluster_path)
                centers_k = centers[..., :12]
                centers_v = centers[..., 12:]
                centers_week = torch.load(cluster_week_path)
                centers_k_week = centers_week[..., :12]
                centers_v_week = centers_week[..., 12:]
                self.t_n = centers_k.shape[-2]
                return centers_k, centers_v, centers_k_week, centers_v_week
        elif '0' in self.args.model_comment or 'JiNan' in self.args.model_comment:
            # cluster_path = f'dataset/PEMS/{self.args.model_comment}_cluster={self.t_n}_few={self.args.few_shot}.pt'
            cluster_path = f'dataset/PEMS/{self.args.model_comment}_cluster={self.t_n}.pt'
            if os.path.exists(cluster_path):
                centers = torch.load(cluster_path)
                centers_k = centers[..., :12]
                centers_v = centers[..., 12:]
                self.t_n = centers_k.shape[-2]
                return centers_k, centers_v, 0, 0
        else:
            cluster_path = f'dataset/PEMS/{self.args.model_comment}_cluster={self.t_n}_min.pt'
            print('cluster_path:', cluster_path)
            if os.path.exists(cluster_path):
                centers = torch.load(cluster_path)
                centers_k = centers[..., :12]
                centers_v = centers[..., 12:]
                self.t_n = centers_k.shape[-2]
                return centers_k, centers_v, 0, 0

        data = data.swapaxes(0,1)
        N, T = data.shape

        num_splits = (T - 23) // 288
        sub_tensors = []
        for i in range(num_splits):
            start_index = i * 288
            end_index = (i + 1) * 288 + 23
            sub_tensor = data[:, start_index:end_index]
            sub_tensors.append(sub_tensor[:, np.newaxis, :])
        sub_tensors = np.concatenate(sub_tensors, axis=1)     # [N, M, 288+23]
        print('sub_tensors:', sub_tensors.shape)

        day_tensors = []
        for i in range(288):
            day_tensors.append(sub_tensors[:, :, i:i+24][:, np.newaxis, :, :])
        day_tensors = np.concatenate(day_tensors, axis=1)     # [N, 288, M, 24]
        print('day_tensor:', day_tensors.shape)

        if self.t_n>0:
            print(f'obtain clusters, tot {N}:')
            centers = []
            for i in range(N):
                print(f'i={i}')
                times_centers = []
                for j in range(288):
                    kmeans = KMeans(n_clusters=self.t_n, random_state=0)
                    kmeans.fit(day_tensors[i,j,...])
                    times_centers.append(torch.from_numpy(kmeans.cluster_centers_).unsqueeze(dim=0))
                times_centers = torch.cat(times_centers, dim=0)
                centers.append( times_centers.unsqueeze(dim=0) )
            centers = torch.cat(centers, dim=0)
        else:
            centers = torch.from_numpy(day_tensors)
            self.t_n = num_splits
        torch.save(centers, cluster_path)
        print('centers.shape:', centers.shape)
        return centers[..., :12], centers[..., 12:], 0, 0

    
    def get_attscore(self, q, k, node=-1):# q [B, N, T], k [B, N, Y, T]
        
        t_n = k.shape[-2]
        q_expanded = q.unsqueeze(2).expand(-1, -1, t_n, -1)    # [B, N, Y, T]
        diff = (q_expanded - k).abs()  
        square_sum = (diff ** 2).sum(dim=-1)        # [B,N,Y]
        distance = torch.sqrt(square_sum)

        mean_distance = distance.mean(dim=-1, keepdim=True)
        std_distance = distance.std(dim=-1, keepdim=True) + 1e-6
        if distance.shape[-1]==1: std_distance = torch.zeros_like(std_distance) + 1e-6

        result =  mean_distance - distance 
        result = result / std_distance * 10
        
        if node>-1: print(f'att score:{result[0,node,...]}')
        return result
        
    def forward(self, enc, x_mark_enc, dec):# enc [B, N, T] ; x_mark_enc [B]
        B, N, T = enc.shape

        Q = enc.clone()
        
        if '0' in self.args.model_comment or 'JiNan' in self.args.model_comment:
            t = x_mark_enc[..., 0]
            k_idx = t.to(torch.int64).reshape(B,1,1,1,1).repeat(1, N, 1, self.t_n, self.seq_len)    # B, N, 1, tn, T

            k = self.k.unsqueeze(dim=0).expand(B, -1, -1, -1, -1)   # B, N, vn, tn, T
            K = torch.gather(input=k, dim=2, index=k_idx).squeeze(dim=2).float() # B, N, tn, T

            v = self.v.unsqueeze(dim=0).expand(B, -1, -1, -1, -1)   # B, N, vn, tn, T
            V = torch.gather(input=v, dim=2, index=k_idx).squeeze(dim=2).float() # B, N, tn, T
        
        else:
            K = self.k.squeeze(dim=1).unsqueeze(dim=0).expand(B, -1, -1, -1).float()          # [N, 1, tn, T] -> [B, N, tn, T]
            V = self.v.squeeze(dim=1).unsqueeze(dim=0).expand(B, -1, -1, -1).float()

        if self.week_opt:
            tw = x_mark_enc[..., 1]*288 + x_mark_enc[..., 0]    # B
            # tw = x_mark_enc[..., 1]
            kw_idx = tw.to(torch.int64).reshape(B,1,1,1,1).repeat(1, N, 1, 2, self.seq_len)    # B, N, 1, tn, T
            kw = self.kw.unsqueeze(dim=0).expand(B, -1, -1, -1, -1)   # B, N, vn, tn, T
            K_week = torch.gather(input=kw, dim=2, index=kw_idx).squeeze(dim=2).float() # B, N, tn, T
            vw = self.vw.unsqueeze(dim=0).expand(B, -1, -1, -1, -1)   # B, N, vn, tn, T
            V_week = torch.gather(input=vw, dim=2, index=kw_idx).squeeze(dim=2).float() # B, N, tn, T
            K = torch.cat([K, K_week], dim=-2)
            V = torch.cat([V, V_week], dim=-2)

        att_score = self.get_attscore(Q, K)
        att_score_softmax = torch.softmax(att_score, dim=-1) # [B, N, Y]
        att_out = torch.einsum('bny,bnyd->bnd', [att_score_softmax, V]) # [B, N, D]

        # 防止意外
        Q_like = ( Q + self.Q_bias ).unsqueeze(dim=-2)
        mean_score = self.get_attscore(Q, torch.cat([K, Q_like],dim=-2) )
        mean_score = torch.softmax(mean_score, dim=-1)

        w = (torch.einsum( 'bny,ny->bn', [mean_score, self.att_weight] ) + self.att_bias.unsqueeze(dim=0)).unsqueeze(dim=-1)    # [B, N, 1]
        out = (1-w) * att_out + w * dec
        # out = att_out

        return out
