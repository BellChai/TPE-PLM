import numpy as np
import scipy.sparse as sp
import pickle
import pandas as pd
import torch
import torch.nn as nn

def pre_graph_dict(args):
    A_dict_np = {}
    A_dict = {}
    lap_dict = {}
    node_dict = {}
    node_dict['PEMS08'], node_dict['PEMS07'], node_dict['PEMS04'], node_dict['PEMS03'] = 170, 883, 307, 358
    for data_graph in args.dataset_graph:
        if data_graph == 'PEMS08' or data_graph == 'PEMS04' or data_graph == 'PEMS07' or data_graph == 'PEMS03':
            A, Distance = get_adjacency_matrix(distance_df_filename='dataset/PEMS/' + data_graph + '.csv',
                                               num_of_vertices=node_dict[data_graph])
        else:
            print('dataset name have error!!!\n\n')
            return
        lpls = cal_lape(A.copy())
        lpls = torch.FloatTensor(lpls).to(args.device)
        lap_dict[data_graph] = lpls
        A = get_normalized_adj(A)
        A_dict_np[data_graph] = A
        A = torch.FloatTensor(A).to(args.device)
        A_dict[data_graph] = A
    args.A_dict_np = A_dict_np
    args.A_dict = A_dict
    args.lpls_dict = lap_dict

def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA



def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    isolated_point_num = np.sum(np.where(d, 0, 1))
    print(f"Number of isolated points: {isolated_point_num}")
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian, isolated_point_num

def cal_lape(adj_mx):
    lape_dim = 32
    L, isolated_point_num = calculate_normalized_laplacian(adj_mx)
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    laplacian_pe = EigVec[:, isolated_point_num + 1: lape_dim + isolated_point_num + 1]
    return laplacian_pe