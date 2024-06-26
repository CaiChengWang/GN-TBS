"""
File: models.py
Author: Caicheng Wang
Description: Implements GN-TBS Architecture
"""

import torch.nn as nn
import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, PReLU
from torch_geometric.utils import scatter
from virtual_node import VirtualNode


##function for message aggregation
def aggregate_incoming_messages(messages, recievers_f, n_total_nodes):
    return scatter(messages, recievers_f, dim=0, dim_size=n_total_nodes, reduce='sum')

##funtion for MLP
def MLP(hidden_layers, input_dim, hidden_dim, embed_dim):
    mlp = Seq()
    for i in range(hidden_layers):
        mlp.add_module('lin' + str(i + 1), Lin(input_dim, hidden_dim))
        mlp.add_module('PReLU' + str(i + 1), PReLU())
        input_dim = hidden_dim
    mlp.add_module('lin' + str(i + 2), Lin(hidden_dim, embed_dim))
    return mlp


## function for building the Edge model
class EdgeModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(EdgeModel, self).__init__()

        # Define an MLP containing two hidden layers with 128 neurons per layer
        self.edge_mlp = Seq(Lin(input_dim * 2 + input_dim, hidden_dim),
                            ReLU(),
                            Lin(hidden_dim, hidden_dim),
                            ReLU(),
                            Lin(hidden_dim, embed_dim),
                            )

        self.edge_mlp.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    def forward(self, edge_attr, sender, receiver):
        out = torch.cat([edge_attr, sender, receiver], dim=1)
        return self.edge_mlp(out)


## function for building the Node model
class NodeModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(NodeModel, self).__init__()

        # Define an MLP containing two hidden layers with 128 neurons per layer
        self.node_mlp = Seq(Lin(input_dim * 3, hidden_dim),
                            ReLU(),
                            Lin(hidden_dim, hidden_dim),
                            ReLU(),
                            Lin(hidden_dim, embed_dim),
                            )

        self.node_mlp.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    def forward(self, x, received_messages, messages_u):
        out = torch.cat([x, received_messages, messages_u], dim=1)
        return self.node_mlp(out)


## function for building the Global model
class GlobalModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(GlobalModel, self).__init__()

        # Define an MLP containing two hidden layers with 128 neurons per layer
        self.global_mlp = Seq(Lin(input_dim * 2, hidden_dim),
                              ReLU(),
                              Lin(hidden_dim, hidden_dim),
                              ReLU(),
                              Lin(hidden_dim, embed_dim),
                              )

        self.global_mlp.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    def forward(self, x, u, batch):
        out = torch.cat([x, u[batch]], dim=1)
        return self.global_mlp(out)


##function for building single step of message passing
class MessagePassingStep(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(MessagePassingStep, self).__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # 定义边更新、节点、全局模型
        self.EdgeModel = EdgeModel(self.input_dim, self.hidden_dim, self.embed_dim)
        self.NodeModel = NodeModel(self.input_dim, self.hidden_dim, self.embed_dim)
        self.GlobalModel = GlobalModel(self.input_dim, self.hidden_dim, self.embed_dim)

    def forward(self, x, senders, receivers, edge_attr, u, batch):

        # Computing messages on edges,
        # which is similar to computing interaction forces between nodes and their neighbors
        messages = self.EdgeModel(edge_attr, x[senders], x[receivers])

        # Computing global messages,
        # which similar to an external force on a node
        messages_u = self.GlobalModel(x, u, batch)

        #aggregate incoming messages m_{sj,rj} from nodes sj to rj
        received_messages_ij = aggregate_incoming_messages(messages, receivers, x.shape[0])

        # aggregate incoming messages m_{rj,sj} from nodes rj to sj
        # m_{sj,rj} = -m_{rj,sj} (momentum conservation property of the message passing)
        received_messages_ji = aggregate_incoming_messages(-messages, senders, x.shape[0])

        received_messages = received_messages_ij + received_messages_ji

        # Compute update of nodes based on global and edge information
        gx = self.NodeModel(x, received_messages, messages_u)

        # return updated node and edge representations with residual connection
        x_update = gx + x
        edge_attr_update = edge_attr + messages
        return x_update, edge_attr_update


##Framework for GN-TBS
# establish the mapping relationships denoted as G^a to ΔV^a
class DeepGraphEmulator(nn.Module):

    def __init__(self, num_x_attr, num_edge_attr, num_u_attr, hidden_dim, embed_dim, hidden_layers, K):
        super(DeepGraphEmulator, self).__init__()
        self.num_x_attr = num_x_attr
        self.num_edge_attr = num_edge_attr
        self.num_u_attr = num_u_attr
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.hidden_layers = hidden_layers
        self.K = K

        ## Initialise internal MLPs:
        # 3 encoder MLPs
        self.node_encode_mlp = MLP(self.hidden_layers, self.num_x_attr, self.hidden_dim, self.embed_dim)
        self.edge_encode_mlp = MLP(self.hidden_layers, self.num_edge_attr, self.hidden_dim,self.embed_dim)
        self.u_encode_mlp = MLP(self.hidden_layers, self.num_u_attr, self.hidden_dim, self.embed_dim)

        # processor MLPs for x-axis,y-axis and z-axis
        self.xx_message_passing_blocks = nn.ModuleList()
        self.yy_message_passing_blocks = nn.ModuleList()
        self.zz_message_passing_blocks = nn.ModuleList()

        # global virtual nodes
        self.vns_xx = nn.ModuleList()
        self.vns_yy = nn.ModuleList()
        self.vns_zz = nn.ModuleList()

        for i in range(self.K):
            ## GN block for x-axis
            # virtual nodes layer
            self.vns_xx.append(VirtualNode(embed_dim, embed_dim, dropout=0))

            # message passing layer
            self.xx_message_passing_blocks.append(MessagePassingStep(
                self.embed_dim,
                self.hidden_dim,
                self.embed_dim))

            # GN block for y-axis
            self.vns_yy.append(VirtualNode(embed_dim, embed_dim, dropout=0))

            # message passing layer
            self.yy_message_passing_blocks.append(MessagePassingStep(
                self.embed_dim,
                self.hidden_dim,
                self.embed_dim))

            # GN block for z-axis
            self.vns_zz.append(VirtualNode(embed_dim, embed_dim, dropout=0))

            # message passing layer
            self.zz_message_passing_blocks.append(MessagePassingStep(
                self.embed_dim,
                self.hidden_dim,
                self.embed_dim))

        # 3 decoder MLPs
        self.dx_decode_mlp = MLP(self.hidden_layers, self.embed_dim, self.hidden_dim, 1)
        self.dy_decode_mlp = MLP(self.hidden_layers, self.embed_dim, self.hidden_dim, 1)
        self.dz_decode_mlp = MLP(self.hidden_layers, self.embed_dim, self.hidden_dim, 1)

    def forward(self, data):

        # input: augmented graph
        x = data.x
        edge_index = data.edge_index
        u = data.u
        batch = data.batch
        real_nodes = data.real_nodes.reshape(-1)
        receivers, senders = edge_index
        edge_attr = data.edge_attr

        ## Encoder:
        # Converting augmented input graph into a more suitable representation for subsequent processor
        x = self.node_encode_mlp(x)
        edge_attr = self.edge_encode_mlp(edge_attr)
        u = self.u_encode_mlp(u)

        ## Processor:
        # Simulating the propagation of forces during tube bending
        xx = yy = zz = x
        xx_edge_attr = yy_edge_attr = zz_edge_attr = edge_attr
        vxx = vyy = vzz = None

        for i in range(self.K):
            ##GN block for X-axis
            xx, vxx = self.vns_xx[i].update_node_emb(xx, edge_index, batch, vx=vxx)
            xx, xx_edge_attr = self.xx_message_passing_blocks[i](xx, senders, receivers,
                                                                 xx_edge_attr, u, batch)
            vxx = self.vns_xx[i].update_vn_emb(xx, batch, vxx)

        for i in range(self.K):
            ##GN block for Y-axis
            yy, vyy = self.vns_yy[i].update_node_emb(yy, edge_index, batch, vx=vyy)
            yy, yy_edge_attr = self.yy_message_passing_blocks[i](yy, senders, receivers,
                                                                 yy_edge_attr, u, batch)
            vyy = self.vns_yy[i].update_vn_emb(yy, batch, vyy)

        for i in range(self.K):
            ##GN block for Z-axis
            zz, vzz = self.vns_zz[i].update_node_emb(zz, edge_index, batch, vx=vzz)
            zz, zz_edge_attr = self.zz_message_passing_blocks[i](zz, senders, receivers,
                                                                 zz_edge_attr, u, batch)
            vzz = self.vns_zz[i].update_vn_emb(zz, batch, vzz)

        ## Decoder:
        # predicting the node’s displacement along each axis
        dx = self.dx_decode_mlp(xx[real_nodes])
        dy = self.dy_decode_mlp(yy[real_nodes])
        dz = self.dz_decode_mlp(zz[real_nodes])

        return torch.cat([dx, dy, dz], 1)