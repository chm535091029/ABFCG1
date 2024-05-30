import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import numpy as np
import torch.nn.functional as F
# from run_gan import *
# import dgl
# from model_gan import *
from torch_geometric.data import Data, Batch


# from data_process import *
class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
        super(GATModel, self).__init__()
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(input_dim, hidden_dim, heads=num_heads))
        self.gat_layers.append(GATConv(hidden_dim * num_heads, output_dim, heads=1))

    def forward(self, data, batch_size):
        x, edge_index = data.x.view(-1, data.x.size(-1)), data.edge_index

        for gat_layer in self.gat_layers:
            x = gat_layer(x.to(torch.float32), edge_index)
            x = nn.ReLU()(x)
        x = F.normalize(x, p=2, dim=-1)
        return x.view(batch_size, -1, x.size(-1))


def get_ast_feature(ast_tensors, ast_node_max, ast_edge_max):
    # feature_list = []
    # print("ast_tensors"+ str(ast_tensors.shape))
    # for batch_i in range(ast_tensors.size(0)):

    ast_node_feature = ast_tensors[:, ast_node_max:]
    adj_matrix = ast_tensors[:, :ast_node_max]  #
    edge_list = []
    # g = dgl.DGLGraph()
    # g.add_nodes(ast_node_max)
    # g.ndata['value'] = torch.tensor(ast_node_feature)
    for row in range(adj_matrix.shape[0]):
        non_zero_indices = np.nonzero(adj_matrix[row])[0]
        if len(non_zero_indices):
            for n in non_zero_indices:
                if len(edge_list) < ast_edge_max:
                    edge_list.append([row, int(n)])
                    # g.add_edges(row,int(n))
    if len(edge_list) < ast_edge_max:
        for _ in range(ast_edge_max - len(edge_list)):
            edge_list.append([0, 0])
    # print(edge_list)
    ast_edge_feature = torch.tensor(edge_list, dtype=torch.long, device="cpu")

    # print("ast_edge_feature"+str(ast_edge_feature.shape))
    ast_edge_feature = ast_edge_feature.transpose(-1, -2).contiguous()
    # data = Data(x=ast_node_feature, edge_index=ast_edge_feature)
    # feature_list.append(data)
    # batch_data = Batch.from_data_list(feature_list)
    return ast_node_feature, ast_edge_feature

