import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
from configloader import Config

config = Config()

#local embedding
class LocalembeddingHeteroGAT(torch.nn.Module):
    def __init__(self):
        super(LocalembeddingHeteroGAT, self).__init__()
        hidden_channels = config.hidden_channels
        output_channels = config.output_channels
        gene_feature_channels = config.gene_feature_channels
        disease_feature_channels = config.disease_feature_channels        
        heads = config.heads

        self.gat1 = HeteroConv({
            ('gene','gene_to_disease_edge','disease'):GATConv((gene_feature_channels,disease_feature_channels),disease_feature_channels,heads=heads,add_self_loops=False),
            ('disease','rev_gene_to_disease_edge','gene'):GATConv((disease_feature_channels,gene_feature_channels),gene_feature_channels,heads=heads,add_self_loops=False),
            ('gene','gene_to_gene_edge','gene'):GATConv(gene_feature_channels,gene_feature_channels,heads=heads,add_self_loops=False),
            ('disease','disease_to_disease_edge','disease'):GATConv(disease_feature_channels,disease_feature_channels,heads=heads,add_self_loops=False),
        },aggr='mean')

        #这里暂时删除了最后一层的多头，以保证输出embedding的维度正确
        #aggr='mean'表示聚合方法为mean
        """
        self.gat2 = HeteroConv({
            ('gene','gene_to_disease_edge','disease'):GATConv((gene_feature_channels,disease_feature_channels),disease_feature_channels,heads=heads,add_self_loops=False),
            ('disease','rev_gene_to_disease_edge','gene'):GATConv((disease_feature_channels,gene_feature_channels),gene_feature_channels,heads=heads,add_self_loops=False),
            ('gene','gene_to_gene_edge','gene'):GATConv(gene_feature_channels,gene_feature_channels,heads=heads,add_self_loops=False),
            ('disease','disease_to_disease_edge','disease'):GATConv(disease_feature_channels,disease_feature_channels,heads=heads,add_self_loops=False),
        },aggr='mean')
        """
        self.gat3 = HeteroConv({
            ('gene','gene_to_disease_edge','disease'):GATConv((gene_feature_channels,disease_feature_channels),disease_feature_channels,heads=heads,add_self_loops=False),
            ('disease','rev_gene_to_disease_edge','gene'):GATConv((disease_feature_channels,gene_feature_channels),gene_feature_channels,heads=heads,add_self_loops=False),
            ('gene','gene_to_gene_edge','gene'):GATConv(gene_feature_channels,gene_feature_channels,heads=heads,add_self_loops=False),
            ('disease','disease_to_disease_edge','disease'):GATConv(disease_feature_channels,disease_feature_channels,heads=heads,add_self_loops=False),
        },aggr='mean')
        
        self.lin1 = Linear(gene_feature_channels*heads, gene_feature_channels)
        self.lin2 = Linear(disease_feature_channels*heads, disease_feature_channels)
        """
        self.lin3 = Linear(gene_feature_channels*heads, gene_feature_channels)
        self.lin4 = Linear(disease_feature_channels*heads, disease_feature_channels)
        """
        self.lin5 = Linear(gene_feature_channels*heads, 256)
        self.lin6 = Linear(disease_feature_channels*heads, 256)

        self.lin7 = Linear(256, 128)
        self.lin8 = Linear(256, 128)
        
    def forward(self,data):
                          
        x_dict = self.gat1(data.x_dict,data.edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict['gene'] = self.lin1(x_dict['gene'])
        x_dict['disease'] = self.lin2(x_dict['disease'])
        """
        x_dict = {key:F.dropout(x, training=self.training) for key, x in x_dict.items()}
        x_dict = self.gat2(x_dict,data.edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict['gene'] = self.lin3(x_dict['gene'])
        x_dict['disease'] = self.lin4(x_dict['disease'])
        """
        x_dict = {key:F.dropout(x, training=self.training) for key, x in x_dict.items()}
        x_dict = self.gat3(x_dict,data.edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict['gene'] = self.lin5(x_dict['gene'])
        x_dict['disease'] = self.lin6(x_dict['disease'])
        
        x_dict['gene'] = self.lin7(x_dict['gene'])
        x_dict['disease'] = self.lin8(x_dict['disease'])
        return x_dict
      
#这里暂时删除了最后一层的多头，以保证输出embedding的维度正确

#local reinforcement GNN
class LrGNN(torch.nn.Module):
    def __init__(self):
        super(LrGNN, self).__init__()
        self.LBGATH = LocalembeddingHeteroGAT()
        self.LBGATH2 = LocalembeddingHeteroGAT()
        self.LBGATH3 = LocalembeddingHeteroGAT()

        self.lin1 = Linear(128*3, 256)
        self.lin2 = Linear(128*3, 256)

        self.lin3 = Linear(256, 128)
        self.lin4 = Linear(256, 128)
        #self.w = torch.nn.Parameter(torch.zeros(hidden_channels,hidden_channels,dtype=torch.float).to(device))
    def forward(self,data_go,data_seq,data_sim):
        go_x_dict = self.LBGATH(data_go)
        sep_x_dict = self.LBGATH2(data_seq)
        sim_x_dict = self.LBGATH3(data_sim)
        disease = torch.cat((go_x_dict['disease'], sep_x_dict['disease']),dim=1)
        disease = torch.cat((disease,sim_x_dict['disease']),dim=1)

        gene = torch.cat((go_x_dict['gene'], sep_x_dict['gene']),dim=1)
        gene = torch.cat((gene,sim_x_dict['gene']),dim=1)

        gene = self.lin1(gene)
        disease = self.lin2(disease)

        gene = self.lin3(gene)
        disease = self.lin4(disease)

        predict = torch.mm(gene,disease.t())
        predict = torch.sigmoid(predict)
        #predict = torch.clamp(predict,0.0001,0.9999)
        #print('sigmoid处理过的predict:',predict)
        #predict = self.bn(predict)
        #predict = predict.relu()

        return predict