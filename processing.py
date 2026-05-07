import scipy.sparse as sp
import numpy as np
import random
import copy
import torch
import math
from configloader import Config

config = Config()

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def network_edge_threshold(network_adj, threshold):
    edge_tmp, edge_value, shape_tmp = sparse_to_tuple(network_adj)
    preserved_edge_index = np.where(edge_value>threshold)[0]
    preserved_network = sp.csr_matrix(
        (edge_value[preserved_edge_index], 
        (edge_tmp[preserved_edge_index,0], edge_tmp[preserved_edge_index, 1])),
        shape=shape_tmp)
    return preserved_network


def search_array(array,aim):
    result = []
    for index in range(0,len(array)):
        if array[index] == aim:
            result.append(index)
    return np.array(result)

"""
def data_mask(data):

    row_g_d, col_g_d = data.edge_index_dict['gene', 'gene_to_disease_edge', 'disease']
    row_d_g, col_d_g = data.edge_index_dict['disease', 'rev_gene_to_disease_edge', 'gene']

    data['gene', 'gene_to_disease_edge', 'disease'].edge_index = None
    data['disease', 'rev_gene_to_disease_edge', 'gene'].edge_index = None

    num_t = config.training_mask
    # Positive edges.
    perm = torch.randperm(row_g_d.size(0))
    row_g_d, col_g_d = row_g_d[perm], col_g_d[perm]
    row_d_g, col_d_g = row_d_g[perm], col_d_g[perm]

    #r, c = row_g_d[:num_t], col_g_d[:num_t]
    #data.val_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row_g_d[num_t:], col_g_d[num_t:]
    data['gene', 'gene_to_disease_edge', 'disease'].edge_index = torch.stack([r, c], dim=0)

    #r, c = row_d_g[:num_t], col_d_g[:num_t]
    #data.val_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row_d_g[num_t:], col_d_g[num_t:]
    data['disease', 'rev_gene_to_disease_edge', 'gene'].edge_index = torch.stack([r, c], dim=0)

    return data
"""