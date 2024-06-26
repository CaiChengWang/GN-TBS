"""
File: tube2graph.py
Author: Caicheng Wang
Description: construct the tube dataset, transform the raw tube data into graph
"""
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import warnings
from torch_geometric.datasets import TUDataset

warnings.filterwarnings("ignore", category=Warning)


class TubeDataset(InMemoryDataset):
    def __init__(self, root, pos_transform, transform=None, pre_transform=None):
        self.pos_transform = pos_transform
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        print(self.data)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['datas.pt']

    def download(self):
        pass

    # def position_transform(self, nodes, BA, R0):
    #
    #     # feature augmentation
    #     lb = BA * R0
    #     y = nodes.loc[:, 'y']
    #     z = nodes.loc[:, 'z']
    #
    #     # Initialize y,z
    #     yt = np.zeros(len(y))
    #     zt = np.zeros(len(z))
    #
    #     # partition 1:
    #     p1 = (nodes.loc[:, 'if_LT'] == 1) + (nodes.loc[:, 'if_LB'] == 1)
    #     alpha_1 = np.pi - BA - np.arctan(z / y)
    #     yt[p1] = (- np.sqrt(y ** 2 + z ** 2) * np.cos(alpha_1))[p1]
    #     zt[p1] = (np.sqrt(y ** 2 + z ** 2) * np.sin(alpha_1))[p1]
    #
    #     # partition 2:
    #     p2 = (nodes.loc[:, 'if_MT'] == 1) + (nodes.loc[:, 'if_MB'] == 1)
    #     alpha_2 = BA + z / R0
    #     yt[p2] = (y * np.cos(alpha_2))[p2]
    #     zt[p2] = (y * np.sin(alpha_2))[p2]
    #
    #     # partition 3:
    #     p3 = (nodes.loc[:, 'if_RT'] == 1) + (nodes.loc[:, 'if_RB'] == 1)
    #     yt[p3] = y[p3]
    #     zt[p3] = (z + lb)[p3]
    #
    #     nodes['y'] = yt
    #     nodes['z'] = zt
    #     return nodes

    # def x_attr(self, nodes, pos_transform, BA, R0):
    #
    #     node = nodes.copy()
    #
    #     y = np.array(node.loc[:, 'y']) - R0
    #     x = np.array(node.loc[:, 'x'])
    #
    #     rp = np.sqrt(y ** 2 + x ** 2)
    #     theta = (np.arcsin(np.abs(y) / rp) * 180) / np.pi
    #     theta_360 = ((x >= 0) * (y >= 0)) * theta + ((x < 0) * (y >= 0)) * (180 - theta) + ((x < 0) * (y < 0)) * (
    #                 180 + theta) + ((x >= 0) * (y < 0)) * (360 - theta)
    #
    #     # feature augmentation
    #     if pos_transform == True: node = self.position_transform(node, BA, R0)
    #
    #     node = np.array(node)
    #     yy = node[:, 8]
    #     zz = node[:, 9]
    #     Rp = np.sqrt(yy ** 2 + zz ** 2)
    #
    #     beta = (np.arccos(node[:, 8] / Rp) * 180) / np.pi
    #     beta = (zz < 0) * (-beta) + (zz >= 0) * beta
    #     nodes_attr = np.concatenate([node[:, [7, 8, 9]], theta.reshape(-1, 1), Rp.reshape(-1, 1), beta.reshape(-1, 1)],
    #                                 axis=1)
    #
    #     return nodes_attr, theta_360

    # def concat_virtual_nodes(self, nodes, edge_index):
    #     # Structural augmentation
    #     # Get the number of boundary nodes
    #     node = nodes.copy()
    #     edge_z = node.loc[0, 'z']
    #     n_edge = (node.loc[:, 'z'] == edge_z).sum()
    #
    #     # Number of nodes
    #     n = len(node)
    #
    #     delta_z = node.loc[0, 'z'] - node.loc[n_edge, 'z']
    #
    #     # Add virtual nodes for left boundary
    #     left_virtual_nodes = node.iloc[0:n_edge, :].copy()
    #     left_virtual_nodes.loc[:, 'z'] += delta_z
    #
    #     # Add virtual nodes for right boundary
    #     right_virtual_nodes = node.iloc[n - n_edge:, :].copy()
    #     right_virtual_nodes.loc[:, 'z'] -= delta_z
    #
    #     # Concatenate virtual nodes and real nodes
    #     concat_nodes = pd.concat([node, left_virtual_nodes, right_virtual_nodes], axis=0, ignore_index=True)
    #
    #     # Virtual edges
    #     # Left boundary
    #     # Transversal edge
    #     left_vr1 = np.arange(n, n + n_edge).reshape(1, -1)
    #     left_vs1 = np.arange(0, n_edge).reshape(1, -1)
    #     # Vertical edge
    #     left_vr2 = np.arange(n, n + n_edge).reshape(1, -1)
    #     left_vs2 = np.roll(left_vr2, -1)
    #
    #     # Virtual edges
    #     # Right boundary
    #     # Transversal edge
    #     right_vr1 = np.arange(n - n_edge, n).reshape(1, -1)
    #     right_vs1 = np.arange(n + n_edge, n + 2 * n_edge).reshape(1, -1)
    #     # Vertical edge
    #     right_vr2 = np.arange(n + n_edge, n + 2 * n_edge).reshape(1, -1)
    #     right_vs2 = np.roll(right_vr2, -1).reshape(1, -1)
    #
    #     vr = np.concatenate([left_vr1, left_vr2, right_vr1, right_vr2], axis=1)
    #     vs = np.concatenate([left_vs1, left_vs2, right_vs1, right_vs2], axis=1)
    #
    #     v_edge_index = np.concatenate([vr, vs], axis=0)
    #
    #     # save unidirectional edges
    #     edge_index = np.concatenate([edge_index, v_edge_index], axis=1)
    #
    #     return concat_nodes, edge_index, n
    #
    # def get_edge_index(self, nodes):
    #
    #     # Number of nodes in the tube section
    #     zmin = nodes.loc[0, 'z']
    #     n_circle = (nodes.loc[:, 'z'] == zmin).sum()
    #     num_nodes = len(nodes)
    #     # Number of axial nodes of the tube
    #     n_len = num_nodes / n_circle
    #     horizon_r_index = np.arange(0, num_nodes - n_circle).reshape(1, -1)
    #     horizon_s_index = np.arange(n_circle, num_nodes).reshape(1, -1)
    #     edge_indexs = np.concatenate([horizon_r_index, horizon_s_index], axis=0)
    #     for i in range(int(n_len)):
    #         vertical_r_index = np.arange(n_circle * i, n_circle * (i + 1)).reshape(1, -1)
    #         vertical_s_index = np.roll(vertical_r_index, -1)
    #         edge_index = np.concatenate([vertical_r_index, vertical_s_index], axis=0)
    #         edge_indexs = np.concatenate([edge_indexs, edge_index], axis=1)
    #     return edge_indexs
    #
    # def get_edge_attr(self, edge_index, x, theta):
    #     delta = np.abs(x[edge_index[0, :]] - x[edge_index[1, :]])
    #     delta[:, 3] = np.abs(theta[edge_index[0, :]] - theta[edge_index[1, :]])
    #     delta[:, 3] = (delta[:, 3] > 90) * (360 - delta[:, 3]) + (delta[:, 3] <= 90) * delta[:, 3]
    #     edge_attr = np.concatenate([delta], axis=1)
    #     return edge_attr
    #
    # def get_cross(self, nodes):
    #     z = nodes.loc[:, 'z']
    #     _, ci = np.unique(np.array(-z), return_inverse=True)
    #     return ci
    def process(self):
        data_list = []
        # data_range = list(range(500, 1300))
        # bad_sampes = np.array(pd.read_csv('bad_samples.csv', header=None))
        #
        # # Remove samples with severe distortions
        # for bad_sampe in bad_sampes: data_range.remove(bad_sampe)
        #
        # for i in data_range:
        #     pos = i
        #
        #     # Read raw data of nodes,edge,graph
        #     nodes = pd.read_csv(r'.\raw_data\tube\input\nodes' + str(i) + r'.csv', index_col=0)
        #     graph = pd.read_csv(r'.\raw_data\tube\input\graph' + str(i) + r'.csv', index_col=0)
        #
        #     # Read the raw_data of the node after the springback
        #     bent_pos = pd.read_csv(r'.\raw_data\tube\output\bent_pos' + str(i) + r'.csv', index_col=0)
        #     edge_index = self.get_edge_index(nodes)
        #
        #     # structural augmentation
        #     nodes, edge_index, num_real_nodes = self.concat_virtual_nodes(nodes, edge_index)
        #
        #     # Get section index
        #     ci = self.get_cross(nodes)
        #
        #     pos_transform = True
        #     BA = graph.loc[0, 'BA'] * np.pi / 180
        #     R0 = graph.loc[0, 'd0'] * graph.loc[0, 'R0/d0']
        #     x, theta = self.x_attr(nodes, pos_transform, BA, R0)
        #
        #     edge_index = edge_index[:, np.argsort(edge_index[0, :])]
        #     edge_attr = self.get_edge_attr(edge_index, x, theta)
        #
        #     edge_index = torch.tensor(edge_index, dtype=torch.long)
        #     edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        #     x = torch.tensor(x, dtype=torch.float)
        #     u = torch.tensor(np.array(graph), dtype=torch.float)
        #     y = torch.tensor(np.array(bent_pos.loc[:, ['xx', 'yy', 'zz']]), dtype=torch.float)
        #     ci = torch.tensor(ci, dtype=torch.int)
        #     real_nodes = torch.tensor((np.arange(0, len(x)) < num_real_nodes), dtype=torch.bool).reshape(-1, 1)
        #
        #     y = y - x[real_nodes.reshape(-1)][:, [0, 1, 2]]
        #
        #     data = Data(x=x, edge_index=edge_index.contiguous(), edge_attr=edge_attr,
        #                 y=y, u=u, pos=pos, num_real_nodes=num_real_nodes,
        #                 real_nodes=real_nodes, ci=ci)
        #
        #     data_list.append(data)
        #     print(i)
        #
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        #
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        #
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
if __name__ == '__main__':
    TubeDataset(root="Tube_data", pos_transform=True)
