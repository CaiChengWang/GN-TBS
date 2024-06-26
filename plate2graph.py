"""
File: plate2graph.py
Author: Caicheng Wang
Description: construct the plate dataset, transform the raw tube data into graph
"""
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import warnings

warnings.filterwarnings("ignore", category=Warning)


class PlateDataset(InMemoryDataset):
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

    # def position_transform(self, nodes, D, t):
    #     #feature augmentation
    #     theta = 2 * np.arctan(7 / D)
    #     x = nodes.loc[:, 'x']
    #     y = nodes.loc[:, 'y']
    #
    #     xt = (x > 0) * (-x * np.cos(theta)) + (x <= 0) * x
    #     yt = (x > 0) * (x * np.sin(theta) + t / 2) + (x <= 0) * y
    #
    #     nodes['x'] = xt
    #     nodes['y'] = yt
    #     return nodes

    # def x_attr(self, nodes, pos_transform, D, t):
    #
    #     # feature augmentation
    #     if pos_transform == True: nodes = self.position_transform(nodes, D, t)
    #     nodes_attr = np.array(nodes)
    #
    #     return nodes_attr

    # def concat_virtual_nodes(self, nodes, edge_index):
    #
    #     #structural augmentation
    #
    #     xmin = np.min(nodes.loc[:, 'x'])
    #     xmax = np.max(nodes.loc[:, 'x'])
    #     zmin = np.min(nodes.loc[:, 'z'])
    #     zmax = np.max(nodes.loc[:, 'z'])
    #
    #     x_max_index = np.where(nodes.loc[:, 'x'] == xmax)[0]
    #     x_min_index = np.where(nodes.loc[:, 'x'] == xmin)[0]
    #     z_max_index = np.where(nodes.loc[:, 'z'] == zmax)[0]
    #     z_min_index = np.where(nodes.loc[:, 'z'] == zmin)[0]
    #
    #     x_max_index_sorted = x_max_index[np.argsort(nodes.loc[x_max_index, 'z'])]
    #     x_min_index_sorted = x_min_index[np.argsort(-nodes.loc[x_min_index, 'z'])]
    #     z_max_index_sorted = z_max_index[np.argsort(-nodes.loc[z_max_index, 'x'])]
    #     z_min_index_sorted = z_min_index[np.argsort(nodes.loc[z_min_index, 'x'])]
    #
    #     sorted_x = np.unique(np.sort(nodes.loc[:, 'x']))
    #     dx = np.abs(sorted_x[0] - sorted_x[1])
    #     x_max_attr = nodes.loc[x_max_index_sorted, :] + [dx, 0, 0]
    #     x_min_attr = nodes.loc[x_min_index_sorted, :] - [dx, 0, 0]
    #
    #     sorted_z = np.unique(np.sort(nodes.loc[:, 'z']))
    #     dz = np.abs(sorted_z[0] - sorted_z[1])
    #     z_max_attr = nodes.loc[z_max_index_sorted, :] + [0, 0, dz]
    #     z_min_attr = nodes.loc[z_min_index_sorted, :] - [0, 0, dz]
    #
    #     corner_index1 = x_max_index_sorted[np.where(x_max_attr.loc[:, 'z'] == zmax)[0]]
    #     corner_index2 = x_max_index_sorted[np.where(x_max_attr.loc[:, 'z'] == zmin)[0]]
    #     corner_index3 = x_min_index_sorted[np.where(x_min_attr.loc[:, 'z'] == zmax)[0]]
    #     corner_index4 = x_min_index_sorted[np.where(x_min_attr.loc[:, 'z'] == zmin)[0]]
    #
    #     corner_attr1 = nodes.loc[corner_index1, :] + [dx, 0, dz]
    #     corner_attr2 = nodes.loc[corner_index2, :] + [dx, 0, -dz]
    #     corner_attr3 = nodes.loc[corner_index3, :] + [-dx, 0, dz]
    #     corner_attr4 = nodes.loc[corner_index4, :] + [-dx, 0, -dz]
    #
    #     concat_nodes = pd.concat(
    #         [nodes, corner_attr1, z_max_attr, corner_attr3, x_min_attr, corner_attr4, z_min_attr, corner_attr2,
    #          x_max_attr], axis=0, ignore_index=True)
    #
    #     n = len(nodes)
    #     nv = len(concat_nodes)
    #
    #     # Virtual edges
    #     cricle_vr = np.arange(n, nv)
    #     cricle_vs = np.roll(cricle_vr, 1)
    #
    #     zmax_vr = np.arange(n + 1, n + 1 + len(z_max_attr))
    #     xmin_vr = np.arange(n + 2 + len(z_max_attr), n + 2 + len(z_max_attr) + len(x_min_attr))
    #     zmin_vr = np.arange(n + 3 + len(z_max_attr) + len(x_min_attr),
    #                         n + 3 + len(z_max_attr) + len(x_min_attr) + len(z_min_attr))
    #     xmax_vr = np.arange(n + 4 + len(z_max_attr) + len(x_min_attr) + len(z_min_attr),
    #                         n + 4 + len(z_max_attr) + len(x_min_attr) + len(z_min_attr) + len(x_max_attr))
    #
    #     zmax_vs = z_max_index_sorted
    #     xmin_vs = x_min_index_sorted
    #     zmin_vs = z_min_index_sorted
    #     xmax_vs = x_max_index_sorted
    #
    #     vr = np.concatenate([cricle_vr, zmax_vr, xmin_vr, zmin_vr, xmax_vr]).reshape(1,-1)
    #     vs = np.concatenate([cricle_vs, zmax_vs, xmin_vs, zmin_vs, xmax_vs]).reshape(1,-1)
    #
    #     virtual_edge_index = np.concatenate([vr,vs],axis=0)
    #
    #     concat_edge_index = np.concatenate([edge_index,virtual_edge_index,np.flip(virtual_edge_index,axis=0)],axis=1)
    #
    #     return concat_nodes, concat_edge_index, n
    #
    # def get_edge_attr(self,nodes,edge_index):
    #     delta = np.array(nodes.loc[edge_index[0,:],['x','y','z']])-np.array(nodes.loc[edge_index[1,:],['x','y','z']])
    #     edge_attr = np.concatenate([delta],axis=1)
    #     return edge_attr

    def process(self):
        data_list = []

        # data_range = list(range(0, 150))
        # for i in data_range:
        #     pos = i
        #     # Read raw data of nodes,edge,graph
        #     nodes = pd.read_csv(r'.\raw_data\plate\input\nodes' + str(i) + r'.csv', index_col=0)
        #     edges = pd.read_csv(r'.\raw_data\plate\input\edges' + str(i) + r'.csv', index_col=0)
        #     graph = pd.read_csv(r'.\raw_data\plate\input\graph' + str(i) + r'.csv', index_col=0)
        #
        #     # Read the raw_data of the node after the springback
        #     bent_pos = pd.read_csv(r'.\raw_data\plate\output\bent_pos' + str(i) + r'.csv', index_col=0)
        #     edge_index = np.array(edges.loc[:, ['receiver', 'sender']]).T
        #
        #     # structural augmentation
        #     nodes, edge_index, num_real_nodes = self.concat_virtual_nodes(nodes, edge_index)
        #
        #     D = graph.loc[0, 'D']
        #     t = graph.loc[0, 't']
        #     x = self.x_attr(nodes, self.pos_transform, D, t)
        #
        #     edge_index = edge_index[:,np.argsort(edge_index[0,:])]
        #     edge_attr = self.get_edge_attr(nodes,edge_index)
        #
        #     edge_index = torch.tensor(edge_index, dtype=torch.long)
        #     edge_attr = torch.tensor(edge_attr,dtype=torch.float)
        #
        #     x = torch.tensor(x, dtype=torch.float)
        #     u = torch.tensor(np.array(graph), dtype=torch.float)
        #     y = torch.tensor(np.array(bent_pos.loc[:, ['xx', 'yy', 'zz']]), dtype=torch.float)
        #
        #     real_nodes = torch.tensor((np.arange(0, len(x)) < num_real_nodes), dtype=torch.bool).reshape(-1, 1)
        #
        #     y = y - x[real_nodes.reshape(-1)]
        #
        #     data = Data(x=x, edge_index=edge_index.contiguous(), edge_attr=edge_attr, y=y, u=u, pos=pos,
        #                 num_real_nodes=num_real_nodes, real_nodes=real_nodes)
        #
        #     data_list.append(data)
        #     print(i)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == '__main__':
    PlateDataset(root="Plate_data",pos_transform=True)
