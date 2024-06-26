"""
File: utils.py
Author: Caicheng Wang
Description: Utility functions for initialising simulator and saving results
"""

import os
import models
import torch
import tube2graph
import plate2graph
import GraphSAGE_GAT
import GN_CM

def create_config_dict(model_type, num_x_attr, num_edge_attr, num_u_attr, hidden_dim, embed_dim, hidden_layers,
                       dataset_name, K, n_epochs, lr,
                       random_seed, batch_size):
    """Creates dictionary of configuration details"""
    return {
        'model_type': model_type,
        'num_x_attr': num_x_attr,
        'num_edge_attr': num_edge_attr,
        'num_u_attr': num_u_attr,
        'hidden_dim': hidden_dim,
        'embed_dim': embed_dim,
        'hidden_layers': hidden_layers,
        'dataset_name,': dataset_name,
        'K': K,
        'n_train_epochs': n_epochs,
        'lr': lr,
        'random_seed': random_seed,
        'batch_size': batch_size,
    }


def create_savedir(results_path, dataset_name, model_type, n_epochs, lr, K, random_seed, batch_size, dir_label):
    """Create directory where simulation results are saved"""
    save_dir = f'emulationResults/{results_path}/{dataset_name}_{model_type}_{n_epochs}epoch_lr{lr}_K{K}_rngseed{random_seed}_batch{batch_size}{dir_label}/'
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    return save_dir


def get_dataset(dataset_name):
    random_seed = 84
    torch.manual_seed(random_seed)
    assert dataset_name == 'tube' or dataset_name == 'plate', 'Unknown dataset'

    if dataset_name == 'tube':
        dataset = tube2graph.TubeDataset(root="Tube_data", pos_transform=True)
    elif dataset_name == 'plate':
        dataset = plate2graph.PlateDataset(root="Plate_data", pos_transform=True)

    dataset = dataset.shuffle()

    #The dataset is diveded into training set, validation set and test set with a radio of 8:1:1
    num_dataset = len(dataset)
    train_dataset = dataset[:int(0.8 * num_dataset)]
    val_dataset = dataset[int(0.8 * num_dataset):int(0.9 * num_dataset)]
    test_dataset = dataset[int(0.9 * num_dataset):num_dataset]

    return train_dataset, val_dataset, test_dataset


def initialise_emulator(emulator_config_dict: dict):
    # initialise GNN architecture based on configuration details
    if emulator_config_dict['model_type'] == 'GN-TBS':
        model = models.DeepGraphEmulator(num_x_attr=emulator_config_dict['num_x_attr'],
                                         num_edge_attr=emulator_config_dict['num_edge_attr'],
                                         num_u_attr=emulator_config_dict['num_u_attr'],
                                         hidden_dim=emulator_config_dict['hidden_dim'],
                                         embed_dim=emulator_config_dict['embed_dim'],
                                         hidden_layers=emulator_config_dict['hidden_layers'],
                                         K=emulator_config_dict['K']
                                         )
    elif emulator_config_dict['model_type'] == 'GraphSAGE':
        model = GraphSAGE_GAT.GNNStack(num_x_attr=emulator_config_dict['num_x_attr'],
                                       num_u_attr=emulator_config_dict['num_u_attr'],
                                       hidden_dim=emulator_config_dict['hidden_dim'],
                                       embed_dim=emulator_config_dict['embed_dim'],
                                       hidden_layers=emulator_config_dict['hidden_layers'],
                                       K=emulator_config_dict['K'],
                                       model_type=emulator_config_dict['model_type'],
                                       heads=1,
                                       )
    elif emulator_config_dict['model_type'] == 'GAT':
        model = GraphSAGE_GAT.GNNStack(num_x_attr=emulator_config_dict['num_x_attr'],
                                       num_u_attr=emulator_config_dict['num_u_attr'],
                                       hidden_dim=emulator_config_dict['hidden_dim'],
                                       embed_dim=emulator_config_dict['embed_dim'],
                                       hidden_layers=emulator_config_dict['hidden_layers'],
                                       K=emulator_config_dict['K'],
                                       model_type=emulator_config_dict['model_type'],
                                       heads=2,
                                       )
    elif emulator_config_dict['model_type'] == 'GN-CM':
        model = GN_CM.DeepGraphEmulator(num_x_attr=emulator_config_dict['num_x_attr'],
                                         num_edge_attr=emulator_config_dict['num_edge_attr'],
                                         num_u_attr=emulator_config_dict['num_u_attr'],
                                         hidden_dim=emulator_config_dict['hidden_dim'],
                                         embed_dim=emulator_config_dict['embed_dim'],
                                         hidden_layers=emulator_config_dict['hidden_layers'],
                                         K=emulator_config_dict['K']
                                         )
    return model

def dataset_zscore_scaler(dataset, y_mean, y_std, x_mean, x_std, edge_attr_mean,edge_attr_std,u_mean,u_std):

    for data in dataset:
        data.x[:] = (data.x[:] - x_mean) / x_std
        data.edge_attr[:] = (data.edge_attr[:] - edge_attr_mean) / (edge_attr_std + 1e-8)
        data.y[:] = (data.y[:] - y_mean) / y_std
        data.u[:] = (data.u[:] - u_mean) / u_std
    return dataset

def reverse_zscore(y, y_mean, y_std):
    return (y * y_std + y_mean)
