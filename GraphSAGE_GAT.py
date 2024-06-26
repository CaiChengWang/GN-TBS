"""
File: GraphSAGE_GAT.py
Author: Caicheng Wang
Description: Model for GraphSAGE and GAT
"""
import torch.nn as nn
from torch_geometric.nn.conv import GATConv,SAGEConv
import torch
from torch.nn import Sequential as Seq, Linear as Lin, PReLU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def MLP(hidden_layers, input_dim, hidden_dim, embed_dim):
    mlp = Seq()
    for i in range(hidden_layers):
        mlp.add_module('lin' + str(i + 1), Lin(input_dim, hidden_dim))
        mlp.add_module('PReLU' + str(i + 1), PReLU())
        input_dim = hidden_dim
    mlp.add_module('lin' + str(i + 2), Lin(hidden_dim, embed_dim))
    return mlp


class GNNStack(torch.nn.Module):
    def __init__(self, num_x_attr,num_u_attr, hidden_dim, embed_dim, hidden_layers, K, model_type, heads):

        super(GNNStack, self).__init__()
        self.K = K

        # Encoder:
        self.node_encode_mlp = MLP(hidden_layers, num_x_attr + num_u_attr, hidden_dim, embed_dim,)

        # Message-passing:
        if model_type == 'GraphSAGE':
            self.convs = nn.ModuleList()
            self.convs.append(SAGEConv(in_channels= embed_dim, out_channels=embed_dim, aggr='sum'))
            assert (K >= 1), 'Number of K is not >=1'
            for l in range(K - 1):
                self.convs.append(SAGEConv(in_channels= embed_dim, out_channels=embed_dim, aggr='sum'))
        elif model_type == 'GAT':
            self.convs = nn.ModuleList()
            self.convs.append(GATConv(in_channels= embed_dim, out_channels=embed_dim))
            assert (K >= 1), 'Number of K is not >=1'
            for l in range(K - 1):
                self.convs.append(GATConv(in_channels= embed_dim, out_channels=embed_dim))

        # Decoder:
        self.node_decode_mlp = MLP(hidden_layers, embed_dim, hidden_dim, 3)

    def forward(self, data):

        edge_index, batch =  data.edge_index, data.batch
        x = data.x
        u = data.u

        #Concatenate x and u
        x = torch.cat([x,u[batch]],dim=1)

        ##Encoder
        x = self.node_encode_mlp(x)

        ##Processor
        for i in range(self.K):
            x = self.convs[i](x, edge_index)

        ##Decoder
        x = self.node_decode_mlp(x[data.real_nodes.reshape(-1)])

        return x
