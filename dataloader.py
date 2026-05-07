import torch
import os
import numpy as np
import scipy.sparse as sp
import random
import sys
import pickle
from configloader import Config
from torch_geometric.data import HeteroData

os.chdir(os.path.dirname(__file__))

config = Config()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DataLoader(object):
    def __init__(self):
        super(DataLoader, self).__init__()
        return
    
    def get_data(self):
        data_go = HeteroData()
        data_seq = HeteroData()
        data_sim = HeteroData()
        if not os.path.exists(config.data_save_path):
            print('The data_save_path do not exist!')
            sys.exit(1)
        with open(config.data_save_path + '/data_go.pkl','rb') as file:
            data_go  = pickle.loads(file.read())            
        with open(config.data_save_path + '/data_seq.pkl','rb') as file:
            data_seq  = pickle.loads(file.read())   
        with open(config.data_save_path + '/data_sim.pkl','rb') as file:
            data_sim  = pickle.loads(file.read())   
        return data_go,data_seq,data_sim


"""
dataloader = DataLoader()
data = dataloader.get_data()
print(data.test_set_coords_dict['test_true'])


n0 = 0
n1 = 0
n_1 = 0
n_2 = 0
test_coords = data.test_set_coords_dict
coo = sp.coo_matrix(data.label)
for i in coo.data:
    if i == 0:
        n0 += 1
    if i == 1:
        n1 += 1
    if i == -1:
        n_1 += 1
    if i == -2:
        n_2 += 1
print(n0)
print(n1)
print(n_1)
print(n_2)
for coord in test_coords['test_true']:
    if data.label[coord[0]][coord[1]] == -1:
        print('True')
    else:
        print('Flase')
for coord in test_coords['test_false']:
    if data.label[coord[0]][coord[1]] == -2:
        print('True')
    else:
        print('Flase')
"""
"""
print(data.one_index)


"""
"""
edge = data.edge_index_dict['disease','disease_to_disease_edge','disease']
coodes = edge.t()
for coode in coodes:
    t = torch.tensor(np.array([coode[1],coode[0]]))
    if t in coodes:
        print(True,coode,t)
"""

"""
coodes = data.edge_index_dict['gene','gene_to_gene_edge','gene'].t()
for coode in coodes:
    #print(coode)
    if coode[0] == coode[1]:
        print(True)

print(data.edge_index_dict['gene','gene_to_gene_edge','gene'])
print(data.edge_index_dict['disease','disease_to_disease_edge','disease'])



dataloader = DataLoader()
data1 = dataloader.get_gene_disease_data()

data2 = HeteroData()

with open(config.data_save_path,'rb') as file:
    gene_disease_data  = pickle.loads(file.read())      

print(data1)
print(data2)
"""

"""
检查项目：
1 矩阵形状
2 矩阵缩略图
3 矩阵非零元素数量
4 矩阵非零元素
"""



"""
data = DataLoader()
gene_disease_data = data.get_gene_disease_data()
print('Step1:检查gene特征矩阵')
print('shape:',gene_disease_data.x_dict['gene'].shape)
print('matrix')
print(gene_disease_data.x_dict['gene'])
gene_feature = sp.coo_matrix(gene_disease_data.x_dict['gene'])
print('nonzero elements number:',len(gene_feature.data))
print('nonzero elements:',gene_feature.data)

print('Step2 检查disease特征矩阵')
print('shape:',gene_disease_data.x_dict['disease'].shape)
print('matrix')
print(gene_disease_data.x_dict['disease'])
disease_feature = sp.coo_matrix(gene_disease_data.x_dict['disease'])
print('nonzero elements number:',len(disease_feature.data))
print('nonzero elements:',disease_feature.data)

print('Step3 检查gene邻接矩阵')
print('shape',gene_disease_data.edge_index_dict['gene','gene_to_gene_edge','gene'].shape)
print('matrix')
print(gene_disease_data.edge_index_dict['gene','gene_to_gene_edge','gene'])
gene_network = sp.coo_matrix(gene_disease_data.edge_index_dict['gene','gene_to_gene_edge','gene'])
print('nonzero elements number:',len(gene_network.data))
print('nonzero elements:',gene_network.data)

print('Step4 检查disease邻接矩阵')
print('shape',gene_disease_data.edge_index_dict['disease','disease_to_disease_edge','disease'].shape)
print('matrix')
print(gene_disease_data.edge_index_dict['disease','disease_to_disease_edge','disease'])
disease_network = sp.coo_matrix(gene_disease_data.edge_index_dict['disease','disease_to_disease_edge','disease'])
print('nonzero elements number:',len(disease_network.data))
print('nonzero elements:',disease_network.data)

print('Step5 检查gene to disease邻接矩阵')
print('shape',gene_disease_data.edge_index_dict['gene','gene_to_disease_edge','disease'].shape)
print('matrix')
print(gene_disease_data.edge_index_dict['gene','gene_to_disease_edge','disease'])
gene_to_disease_network = sp.coo_matrix(gene_disease_data.edge_index_dict['gene','gene_to_disease_edge','disease'])
print('nonzero elements number:',len(gene_to_disease_network.data))
print('nonzero elements:',gene_to_disease_network.data)

print('Step6 检查gene to disease邻接矩阵')
print('shape',gene_disease_data.edge_index_dict['disease','disease_to_gene_edge','gene'])
print('matrix')
print(gene_disease_data.edge_index_dict['disease','disease_to_gene_edge','gene'].shape)
disease_to_gene_network = sp.coo_matrix(gene_disease_data.edge_index_dict['disease','disease_to_gene_edge','gene'])
print('nonzero elements number:',len(disease_to_gene_network.data))
print('nonzero elements:',disease_to_gene_network.data)


val_label = data.get_label()
print('Step7 检查val_label邻接矩阵')
print('shape',val_label.shape)
print('matrix')
print(val_label)
val_label_coo = sp.coo_matrix(val_label)
print('nonzero elements number:',len(val_label_coo.data))
print('nonzero elements:',val_label_coo.data)
if -1 in val_label_coo.data:
    print('-1 in val_label_label')


val_test_set_coords_dict = data.get_val_test_set_coords_dict()
print('Step8 检查val_test_set_coords_dic')
print('val_true')
print(len(val_test_set_coords_dict['val_true']))
print(val_test_set_coords_dict['val_true'])

print('val_false')
print(len(val_test_set_coords_dict['val_false']))
print(val_test_set_coords_dict['val_false'])

print('test_true')
print(len(val_test_set_coords_dict['test_true']))
print(val_test_set_coords_dict['test_true'])

print('test_false')
print(len(val_test_set_coords_dict['test_false']))
print(val_test_set_coords_dict['test_false'])

"""
