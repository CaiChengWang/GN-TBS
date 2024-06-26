'''
The code are adapted from
https://github.com/dodaltuin/passive-lv-gnn-emul/blob/main/models.py
'''

import torch.nn as nn
import torch
from torch.nn import Sequential as Seq, Linear as Lin, Tanh, LayerNorm
from torch_geometric.utils import scatter


def aggregate_incoming_messages(messages, receivers_f, n_total_nodes):
    return scatter(messages, receivers_f, dim=0, dim_size=n_total_nodes, reduce='sum')


def MLP(hidden_layers, input_dim, hidden_dim, embed_dim, norm):
    mlp = Seq()
    for i in range(hidden_layers):
        mlp.add_module('lin' + str(i + 1), Lin(input_dim, hidden_dim))
        mlp.add_module('Tanh' + str(i + 1), Tanh())
        input_dim = hidden_dim
    mlp.add_module('lin' + str(i + 2), Lin(hidden_dim, embed_dim))

    if norm == True:
        mlp.add_module('LayerNorm', LayerNorm(embed_dim))
    return mlp


class EdgeModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(EdgeModel, self).__init__()
        self.edge_mlp = Seq(Lin(input_dim * 2 + input_dim, hidden_dim),
                            Tanh(),
                            Lin(hidden_dim, hidden_dim),
                            Tanh(),
                            Lin(hidden_dim, embed_dim),
                            )
        self.edge_mlp.add_module('LayerNorm', LayerNorm(embed_dim))
        self.edge_mlp.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    def forward(self, edge_attr, sender, receiver):
        out = torch.cat([edge_attr, sender, receiver], dim=1)
        return self.edge_mlp(out)


class NodeModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(NodeModel, self).__init__()
        self.node_mlp = Seq(Lin(input_dim * 2, hidden_dim),
                            Tanh(),
                            Lin(hidden_dim, hidden_dim),
                            Tanh(),
                            Lin(hidden_dim, embed_dim),
                            )
        self.node_mlp.add_module('LayerNorm', LayerNorm(embed_dim))
        self.node_mlp.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    def forward(self, x, received_messages):
        out = torch.cat([x, received_messages], dim=1)
        return self.node_mlp(out)


class MessagePassingStep(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(MessagePassingStep, self).__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.EdgeModel = EdgeModel(self.input_dim, self.hidden_dim, self.embed_dim)
        self.NodeModel = NodeModel(self.input_dim, self.hidden_dim, self.embed_dim)

    def forward(self, x, senders, receivers, edge_attr):
        messages = self.EdgeModel(edge_attr, x[receivers], x[senders])
        # aggregate incoming messages m_{ij} from nodes i to j where i > j
        received_messages_ij = aggregate_incoming_messages(messages, receivers, x.shape[0])

        # aggregate incoming messages m_{ij} from nodes i to j where i < j
        # m_{ij} = -m_{ji} where i < j (momentum conservation property of the message passing)
        received_messages_ji = aggregate_incoming_messages(-messages, senders, x.shape[0])

        # concatenate node representation with incoming messages and then update node representation
        gx = self.NodeModel(x, received_messages_ij + received_messages_ji)
        # return updated node and edge representations with residual connection
        x_update = gx + x
        edge_attr_update = edge_attr + messages
        return x_update, edge_attr_update


## Framework of GN-CM
class DeepGraphEmulator(nn.Module):
    """DeepGraphEmulator for bent tubes"""

    def __init__(self, num_x_attr, num_edge_attr, num_u_attr, hidden_dim, embed_dim, hidden_layers, K):
        super(DeepGraphEmulator, self).__init__()
        self.num_x_attr = num_x_attr
        self.num_edge_attr = num_edge_attr
        self.num_u_attr = num_u_attr
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.hidden_layers = hidden_layers
        self.K = K  # steps of message passing

        ## Initialise internal MLPs:
        # 3 encoder MLPs
        self.node_encode_mlp = MLP(self.hidden_layers, self.num_x_attr, self.hidden_dim, self.embed_dim, norm=True)
        self.edge_encode_mlp = MLP(self.hidden_layers, self.num_edge_attr, self.hidden_dim, self.embed_dim, norm=True)
        self.u_encode_mlp = MLP(self.hidden_layers, self.num_u_attr, self.hidden_dim, self.embed_dim, norm=True)

        # processor MLPs
        self.message_passing_blocks = nn.ModuleList()

        for i in range(self.K):
            # message passing layer
            self.message_passing_blocks.append(MessagePassingStep(
                self.embed_dim,
                self.hidden_dim,
                self.embed_dim))
        # 3 decoder MLPs
        self.dx_decode_mlp = MLP(self.hidden_layers, self.embed_dim * 3, self.hidden_dim, 1, norm=False)
        self.dy_decode_mlp = MLP(self.hidden_layers, self.embed_dim * 3, self.hidden_dim, 1, norm=False)
        self.dz_decode_mlp = MLP(self.hidden_layers, self.embed_dim * 3, self.hidden_dim, 1, norm=False)

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index
        u = data.u
        batch = data.batch
        real_nodes = data.real_nodes.reshape(-1)

        # Filter out edges with sj > rj
        receivers, senders = edge_index
        edge_attr = data.edge_attr
        index_filer = receivers < senders

        edge_attr = edge_attr[index_filer]
        receivers = receivers[index_filer]
        senders = senders[index_filer]

        ## Encoder:
        # encode nodes, edges and u
        x = self.node_encode_mlp(x)
        edge_attr = self.edge_encode_mlp(edge_attr)

        ## Processor:
        # perform K rounds of message passing
        for i in range(self.K):
            x, edge_attr = self.message_passing_blocks[i](x, senders, receivers,
                                                          edge_attr)

        incoming_messages = aggregate_incoming_messages(edge_attr, receivers, x.shape[0])

        z_local = torch.cat([x, incoming_messages], dim=1)

        u = self.u_encode_mlp(u)

        final_representation = torch.cat((u[batch], z_local), dim=1)

        ## Decoder:
        dx = self.dx_decode_mlp(final_representation[real_nodes])
        dy = self.dy_decode_mlp(final_representation[real_nodes])
        dz = self.dz_decode_mlp(final_representation[real_nodes])

        return torch.cat([dx, dy, dz], 1)
