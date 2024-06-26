"""
File: load_model.py
Author: Caicheng Wang
Description: Load the trained model to predict the springback mesh topology
"""
import torch
from torch_geometric.loader import DataLoader
import utils
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd
import time
from torch_geometric.utils import scatter
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#############################################################################
## Set up shell input variables
#############################################################################

parser = argparse.ArgumentParser(description='load model of GNN')
parser.add_argument('--dataset_name', default='tube', type=str, help='Set which dataset to use')
args = parser.parse_args()

#############################################################################
## Set up hard-coded input variables
#############################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Random seed for initialising network parameters
random_seed = 84
torch.manual_seed(random_seed)

#dataset
train_dataset, val_dataset, test_dataset = utils.get_dataset(args.dataset_name)

#z-score
y_mean, y_std = torch.mean(train_dataset[:].y, dim=0), torch.std(train_dataset[:].y, dim=0)
x_mean, x_std = torch.mean(train_dataset[:].x, dim=0), torch.std(train_dataset[:].x, dim=0)
edge_attr_mean, edge_attr_std = torch.mean(train_dataset[:].edge_attr, dim=0), torch.std(train_dataset[:].edge_attr,
                                                                                         dim=0)
u_mean, u_std = torch.mean(train_dataset[:].u, dim=0), torch.std(train_dataset[:].u, dim=0)

train_dataset = utils.dataset_zscore_scaler(train_dataset, y_mean, y_std, x_mean, x_std, edge_attr_mean, edge_attr_std,
                                            u_mean, u_std)
val_dataset = utils.dataset_zscore_scaler(val_dataset, y_mean, y_std, x_mean, x_std, edge_attr_mean, edge_attr_std,
                                          u_mean, u_std)
test_dataset = utils.dataset_zscore_scaler(test_dataset, y_mean, y_std, x_mean, x_std, edge_attr_mean, edge_attr_std,
                                           u_mean, u_std)

#dataloader
train_loader = DataLoader(train_dataset, 1, shuffle=True)
test_loader = DataLoader(test_dataset, 1, shuffle=False)
val_loader = DataLoader(val_dataset, 1, shuffle=False)

#############################################################################
## Set the result reading directory,read the configuration
#############################################################################

results_save_dir = r'.\emulationResults\tubeResults\tube_GN-TBS_3000epoch_lr0.0005_K1_rngseed84_batch16'

# create configuration dictionary of hyperparameters of the GNN emulator
config_file = open(f'{results_save_dir}/config_dict.txt', 'r+')
config_dict = eval(config_file.read())

#############################################################################
## Initialize GNN model
#############################################################################

model = utils.initialise_emulator(config_dict)
model.to(device)

# Define loss
loss = nn.MSELoss(reduction='sum')

# load parameter of model
model_path = f'{results_save_dir}/best_model.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device(device))['model_state_dict'])

#############################################################################
## Test
#############################################################################

model.eval()
sum_loss = 0

total = len(test_dataset)
test_loss = np.zeros((total))
test_error = np.zeros((total))
test_iter = enumerate(test_loader)
t_test = tqdm(total=total, leave=False)

test_list = []
sa_simu = np.zeros((total))
sa_real = np.zeros((total))
time_results = np.zeros((total))

#Calculate the axis slope
def computek(x, y):
    x = x.detach().numpy().reshape(-1, 1)
    y = y.detach().numpy()
    model = LinearRegression()
    model.fit(x, y)
    return model.coef_[0]

#Visualization of axes
def visualize(axis_simu, axis_real):
    axis_simu = axis_simu.detach().numpy()
    x_simu = axis_simu[:, 0]
    y_simu = axis_simu[:, 1]
    z_simu = axis_simu[:, 2]

    axis_real = axis_real.detach().numpy()
    x_real = axis_real[:, 0]
    y_real = axis_real[:, 1]
    z_real = axis_real[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_simu, y_simu, z_simu, c='r', marker='o', label='simu')
    ax.scatter(x_real, y_real, z_real, c='b', marker='o', label='real')

    # Determine the range of axes
    m = (int(int(max([np.max(axis_simu), np.max(axis_real)])) / 10) + 1) * 10
    n = (int(int(min([np.min(axis_simu), np.min(axis_real)])) / 10) + 1) * 10

    # Setting Axis Labels
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')

    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks(range(n, m, 50))
    ax.set_yticks(range(n, m, 50))
    ax.set_zticks(range(n, m, 50))

    # Add Legend
    ax.legend()

    # Display
    plt.show(block=True)

def getspringbackangle(simu_result, real_result, i, ci, BA):
    i = i.detach().item()

    nodes = pd.read_csv(r'.\raw_data\tube\input\nodes' + str(i) + r'.csv', index_col=0)

    axis_simu = scatter(simu_result, ci, dim=0, dim_size=max(ci) + 1, reduce='mean')
    axis_real = scatter(real_result, ci, dim=0, dim_size=max(ci) + 1, reduce='mean')

    # Axis visualization
    visualize(axis_simu, axis_real)

    # Section index to each node of the left and right straight tube segments,
    # only half of which are recorded, since nodes close to bending segments may also have bends
    lindex = ci[int((np.sum(nodes.loc[:, 'if_LT'] + nodes.loc[:, 'if_LB']) - 1) / 2)]
    rindex = ci[len(ci) - int(np.sum(nodes.loc[:, 'if_RT'] + nodes.loc[:, 'if_RB']) / 2)]

    # Axis coordinates of the left straight tube section
    coor_simu_1 = axis_simu[:lindex, [1, 2]]
    coor_real_1 = axis_real[:lindex, [1, 2]]

    # Axis coordinates of the right straight tube section
    coor_simu_2 = axis_simu[rindex:, [1, 2]]
    coor_real_2 = axis_real[rindex:, [1, 2]]

    # Calculate the slope of the left straight tube
    k_simu1 = computek(coor_simu_1[:, 0], coor_simu_1[:, 1])
    k_real1 = computek(coor_real_1[:, 0], coor_real_1[:, 1])

    # Calculate the slope of the right straight tube
    k_simu2 = computek(coor_simu_2[:, 0], coor_simu_2[:, 1])
    k_real2 = computek(coor_real_2[:, 0], coor_real_2[:, 1])

    ABA_real_raw = np.arctan(np.abs((k_real1 - k_real2) / (1 + k_real1 * k_real2))) * 180 / np.pi
    ABA_real = (k_real1 < 0) * ABA_real_raw + (k_real1 >= 0) * (180 - ABA_real_raw)

    ABA_simu_raw = np.arctan(np.abs((k_simu1 - k_simu2) / (1 + k_simu1 * k_simu2))) * 180 / np.pi
    ABA_simu = (k_simu1 < 0) * ABA_simu_raw + (k_simu1 >= 0) * (180 - ABA_simu_raw)

    # Calculate the actual bending angle
    sa_simu = BA - ABA_simu
    sa_real = BA - ABA_real

    return sa_simu, sa_real, axis_simu, axis_real


with torch.no_grad():
    for i, data in test_iter:

        data = data.to(device)
        time_start = time.time()
        batch_out = model(data)
        time_end = time.time()
        time_results[i] = time_end - time_start

        batch_out_reverse = utils.reverse_zscore(batch_out, y_mean, y_std)
        real_out_reverse = utils.reverse_zscore(data.y, y_mean, y_std)

        # compute loss and error
        batch_loss = loss(batch_out_reverse, real_out_reverse)
        batch_loss_item = batch_loss.item()
        sum_loss += batch_loss_item
        test_loss[i] = (batch_loss_item / data.num_real_nodes.sum().item())
        error = torch.sum(torch.sqrt(
            torch.sum((real_out_reverse - batch_out_reverse) ** 2, axis=1))) / data.num_real_nodes.sum().item()
        test_error[i] = error

        simu_x = utils.reverse_zscore(data.x[data.real_nodes.reshape(-1)], x_mean, x_std).detach().numpy()
        simu_result = batch_out_reverse + simu_x[:, [0, 1, 2]]
        real_result = real_out_reverse + simu_x[:, [0, 1, 2]]

        # output loss
        print(f'{data.pos}:loss{batch_loss_item / data.num_real_nodes.sum().item()}')

        # Calculate the springback angle
        u = utils.reverse_zscore(data.u, u_mean, u_std)
        BA = u[0, 10].detach().numpy()
        real_nodes = data.real_nodes.squeeze(-1)
        ci = data.ci[real_nodes].to(torch.long) - 1
        sa_simu[i], sa_real[i], axis_simu, axis_real = getspringbackangle(simu_result, real_result, data.pos, ci, BA)

        # Save the simu result and axis
        simu = np.concatenate([np.arange(1, data.num_real_nodes + 1).reshape(-1, 1), simu_result.detach().numpy()],
                              axis=1)
        np.savetxt(f"{results_save_dir}\simu_{data.pos.detach().item()}.csv", simu, delimiter=',')
        np.savetxt(f"{results_save_dir}/axis_simu_{data.pos.detach().item()}.csv", axis_simu, delimiter=',')
        np.savetxt(f"{results_save_dir}/axis_real_{data.pos.detach().item()}.csv", axis_real, delimiter=',')

print('test_Loss:', sum_loss / val_dataset.num_real_nodes.sum().item())
