import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import warnings
import torch

import scipy.sparse as sp
from scipy.sparse import coo_matrix

warnings.filterwarnings('ignore')

class StandardScaler_cyh():
    def __init__(self):
        self.std_ = 1
        self.mean_ = 0
    def fit(self, x): # tensor [B, N]
        self.mean_ = x.mean()
        self.std_ = x.std()
    def transform(self, x):
        return (x-self.mean_)/self.std_
    def inverse_transform(self, x):
        return x*self.std_+self.mean_

class Dataset_PEMS_graph(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='PEMS08.npz', scale=False, args=None):
        if size == None:
            self.seq_len = 12
            self.label_len = 6
            self.pred_len = 12
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # cyh
        self.node_num = args.node_num
        self.args = args
        self.filter_mode = args.filter_mode
        self.b1_ = args.b1_
        self.b2_ = args.b2_
        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

        if 'PEMS0' in self.data_path:
            self.supports = torch.tensor(self.load_adj(os.path.join(root_path, args.adj_path), self.node_num)).to(torch.double) # list, len=1, supports[0]=tensor(170,170)

    def load_adj(self, adj_file, node_num):
        print('adj_file ', adj_file)
        df = pd.read_csv(adj_file)
        data_raw = df.values
        adj = coo_matrix((data_raw[:,2], (data_raw[:,0], data_raw[:,1])), shape=(node_num, node_num))
        t = adj.todense()
        t = coo_matrix(t)
        adj_lap = self.calculate_normalized_laplacian(t)
        return adj_lap
    
    def calculate_normalized_laplacian(self, adj):
        d = np.array(adj.sum(1))
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        return normalized_laplacian.astype(np.float32).todense()

    def __read_data__(self):
        self.scaler = StandardScaler_cyh()
        np_data = np.load(os.path.join(self.root_path,self.data_path))['data']
        if len(np_data.shape)==3: np_data = np_data[:,:,0]

        self.node_all = np_data.shape[1]
        node_num = self.node_num
        np_data = np_data[:,:node_num]

        entire_len = np_data.shape[0]
        print(f'\nentire_len: {entire_len}, node_num: {node_num}\n')
        b1 = int(entire_len * self.b1_) 
        b2 = int(entire_len * self.b2_)

        border1s = [0, b1, b2]
        border2s = [b1, b2, entire_len]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.set_type==0:
            border2 = int(border2*self.args.few_shot)
            self.args.train_data = np_data[0:border2,:].copy()
            self.args.train_mean = np.mean(self.args.train_data)
            self.args.train_std = np.std(self.args.train_data)

        print(f'np_data shape: {np_data[border1:border2].shape}')
        if self.scale:
            train_data = np_data[border1s[0]:border2s[0]].reshape(-1,1)
            self.scaler.fit(train_data)
            data = self.scaler.transform(np_data.reshape(-1,1)).reshape(-1,node_num)
        else:
            data = np_data

        stamps = torch.Tensor([[i] for i in range(data.shape[0])])
        #                     time of day, day of week,  ...
        # df_stamp = torch.cat([stamps%288, stamps//288%7, stamps%(288*7), stamps%144, stamps%(288*17), stamps], dim=-1)
        df_stamp = torch.cat([stamps%288, stamps//288%7, stamps//12%24, stamps//288%31, stamps//288, stamps], dim=-1)  
        data_stamp = df_stamp[border1:border2]
        print(f'data_stamp: {data_stamp.shape}')

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        print(f"data_x.shape={self.data_x.shape}")
        self.data_stamp = data_stamp

        # 在训练集里创造一些新的数据，模拟聚类中心之外的意外
        if self.set_type==0 and self.args.mode_att_mode!=0:
            # extra_num = int(data.shape[0]*0.07)
            extra_num = int(data.shape[0]*0.07)//288*288 + 288 #* (1 if self.args.cluster>=6 else 0)

            # data_max, data_min = np.max(data[:-288*2,:], axis=0, keepdims=True).swapaxes(0,1), np.min(data[:-288*2, :], axis=0, keepdims=True).swapaxes(0,1)
            if '0' in self.args.dataset:
                data_max, data_min = np.max(data[288*2:,:], axis=0, keepdims=True).swapaxes(0,1), np.min(data[288*2:, :], axis=0, keepdims=True).swapaxes(0,1)
                amplitude = (data_max - data_min)/2
                vertical_shift = amplitude + data_min
                x = np.tile(np.arange(extra_num), (node_num, 1))
                sine_function = amplitude * np.sin(2 * np.pi * (x + (border2-border1)%288 - (72+40) ) / (288)) + vertical_shift + np.random.normal(loc=0, scale=amplitude/20, size=(node_num, extra_num))
                sine_function = sine_function.swapaxes(0,1)
                extra_x = sine_function
                extra_y = sine_function
            else:
                data_max, data_min = np.max(data[288*2:,:], axis=0, keepdims=True).swapaxes(0,1), np.min(data[288*2:, :], axis=0, keepdims=True).swapaxes(0,1)
                line = np.tile( (data_max+data_min)/2 , (extra_num, 1))
                line = np.random.normal(loc=0, scale=line/5, size=(node_num, extra_num)) + line
                extra_x = line
                extra_y = line
            
            # self.data_x = np.concatenate([self.data_x, extra_x], axis=0 )
            # self.data_y = np.concatenate([self.data_y, extra_y], axis=0 )
            self.data_x = np.concatenate([extra_x, self.data_x], axis=0 )
            self.data_y = np.concatenate([extra_y, self.data_y], axis=0 )

            # print(f'extra_num={extra_num}, extra_x.shape={extra_x.shape}, extra_y.shape={extra_y.shape}')

            # stamps = torch.Tensor([[i] for i in range(data.shape[0] + extra_num)])
            # df_stamp = torch.cat([stamps%288, stamps//288%7, stamps//12%24, stamps//288%31, stamps//288, stamps], dim=-1)  
            # data_stamp = df_stamp[:border2 + extra_num]
            # self.data_stamp = data_stamp

            if extra_num<=self.data_stamp.shape[0]:
                self.data_stamp = torch.cat([self.data_stamp[:extra_num], self.data_stamp], dim=0)
            else:
                for i in range(extra_num//288):
                    self.data_stamp = torch.cat([self.data_stamp[:288], self.data_stamp], dim=0)

            # stamps = torch.Tensor([[i] for i in range(data.shape[0] + extra_num )]) #- extra_num
            # df_stamp = torch.cat([stamps%288, stamps//288%7, stamps//12%24, stamps//288%31, stamps//288, stamps], dim=-1)  
            # data_stamp = df_stamp[:border2 + extra_num]
            # self.data_stamp = data_stamp

            print(f'data_stamp: {self.data_stamp.shape}')
            print(f"data_x.shape={self.data_x.shape}")


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        # print(f's_begin: {s_begin}, s_end: {s_end}, r_begin: {r_begin}, r_end: {r_end}')
        seq_x = self.data_x[s_begin:s_end, :]                       # [seq_len, node_num]
        seq_y = self.data_y[r_begin:r_end, :]                       # [pre_len, node_num]
        seq_x_mark = self.data_stamp[s_begin:s_end]                 # [seq_len, 5]
        seq_y_mark = self.data_stamp[r_begin:r_end]                 # [pre_len, 5]

        # print(f'seq_x.shape: {seq_x.shape}, seq_y.shape: {seq_y.shape}, seq_x_mark.shape: {seq_x_mark.shape}, seq_y_mark.shape: {seq_y_mark.shape}')

        return seq_x, seq_y, seq_x_mark, seq_y_mark
        # return seq_y, seq_x, seq_y_mark, seq_x_mark

    def __len__(self):
        return self.tot_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

