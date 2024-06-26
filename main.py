"""
File: main.py
Author: Caicheng Wang
Email: wcc_wy@163.com
Description: Training and Evaluation of axial springback via GNN
https://github.com/CaiChengWang/GN-TBS
"""

# The purpose of program is to simulate the springback of mesh-based tube bending,
# as well as the springback of mesh-based plate V-bending.

# The code for virtual_node references the gtrick.
# gtrick: Bag of Tricks for Graph Neural Networks.
# https://github.com/sangyx/gtrick/blob/main/gtrick/pyg/virtual_node.py

#############################################################################
## Package imports
#############################################################################

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import utils

import argparse

#############################################################################
## Set up shell input variables
# variable values can be changed directly on the command line,
# without having to change them in the source code.

# example: python main.py --learning_rate 0.01
# The above code changes the value of lr directly and run main.py

# Once the simulator is trained, change --mode "evaluate" and re-run
# to evaluate the out of sample predictive performance of the simulator
#############################################################################

parser = argparse.ArgumentParser(description='Simulating axial springback of mesh-based tube bending via GNN')
parser.add_argument('--mode', default='train', choices=['train', 'evaluate'],
                    help='Fit model to training data or evaluate on test data')
parser.add_argument('--model_type', default='GN-TBS', choices=['GN-TBS', 'GraphSAGE', 'GAT', 'GN-CM'],
                    help='Select the type of GNN')
parser.add_argument('--n_epochs', default=3000, type=int, help='Number of epochs to train the model for')
parser.add_argument('--batch_size', default=16, type=int, help='Number of batch size')
parser.add_argument('--dataset_name', default='tube', type=str, help='Set which dataset to use')
parser.add_argument('--results_path', default='tubeResults', type=str,
                    help='Name of sub-directory in "/data" where simulation data is stored')
parser.add_argument('--lr', default=5e-4, type=float, help='Learning rate for training the network')
parser.add_argument('--K', default=1, type=int, help='Number of message passing steps to perform')
parser.add_argument('--read_para', default=False, type=bool,
                    help='Set whether to read the parameters of the previously trained model')
parser.add_argument('--dir_label', default='_', type=str,
                    help='Optional label to append to end of results save directory')
args = parser.parse_args()

#############################################################################
## Set up hard-coded input variables
## hardcode: variables are limited to be modified only in the source code.
#############################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Network architecture parameters
hidden_dim = 128
embed_dim = 128
hidden_layers = 2

# Random seed for initialising network parameters
random_seed = 84
torch.manual_seed(random_seed)

# Dataset acquisition
train_dataset, val_dataset, test_dataset = utils.get_dataset(args.dataset_name)

# Obtaining statistical indicators for data Sets
y_mean, y_std = torch.mean(train_dataset[:].y, dim=0), torch.std(train_dataset[:].y, dim=0)
x_mean, x_std = torch.mean(train_dataset[:].x, dim=0), torch.std(train_dataset[:].x, dim=0)
edge_attr_mean, edge_attr_std = torch.mean(train_dataset[:].edge_attr, dim=0), torch.std(train_dataset[:].edge_attr,
                                                                                         dim=0)
u_mean, u_std = torch.mean(train_dataset[:].u, dim=0), torch.std(train_dataset[:].u, dim=0)

# Z-score standardization
train_dataset = utils.dataset_zscore_scaler(train_dataset, y_mean, y_std, x_mean, x_std, edge_attr_mean, edge_attr_std,
                                            u_mean, u_std)
val_dataset = utils.dataset_zscore_scaler(val_dataset, y_mean, y_std, x_mean, x_std, edge_attr_mean, edge_attr_std,
                                          u_mean, u_std)
test_dataset = utils.dataset_zscore_scaler(test_dataset, y_mean, y_std, x_mean, x_std, edge_attr_mean, edge_attr_std,
                                           u_mean, u_std)

# Building the Dataloader
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

# Number of features
num_x_attr = train_dataset.x.shape[1]
num_edge_attr = train_dataset.edge_attr.shape[1]
num_u_attr = train_dataset.u.shape[1]

#############################################################################
## Set the result save directory,save the configuration
#############################################################################

# Function to create subdir to save simulation results
results_save_dir = utils.create_savedir(results_path=args.results_path,
                                        dataset_name=args.dataset_name,
                                        model_type=args.model_type,
                                        n_epochs=args.n_epochs,
                                        lr=args.lr,
                                        K=args.K,
                                        random_seed=random_seed,
                                        batch_size=args.batch_size,
                                        dir_label=args.dir_label,
                                        )

# Create configuration dictionary of hyperparameters
config_dict = utils.create_config_dict(hidden_dim=hidden_dim,
                                       num_x_attr=num_x_attr,
                                       num_edge_attr=num_edge_attr,
                                       num_u_attr=num_u_attr,
                                       embed_dim=embed_dim,
                                       hidden_layers=hidden_layers,
                                       dataset_name=args.dataset_name,
                                       model_type=args.model_type,
                                       K=args.K,
                                       n_epochs=args.n_epochs,
                                       lr=args.lr,
                                       random_seed=random_seed,
                                       batch_size=args.batch_size)

# write configuration dictionary to a text file in results_save_dir
with open(f'{results_save_dir}/config_dict.txt', 'w') as f: print(str(config_dict), file=f)

#############################################################################
## Initialize GNN model
#############################################################################

model = utils.initialise_emulator(config_dict)
model.to(device)

# Save path of the model
model_path = f'{results_save_dir}/model.pth'
best_model_path = f'{results_save_dir}/best_model.pth'

# Optimizer Settings
optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': args.lr}], args.lr)

# Load the checkpoint model
if args.read_para == True:
    model_path = f'{results_save_dir}/model.pth'
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'], )
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    ini_epoch = checkpoint['epoch'] + 1
else:
    ini_epoch = 0

# Scheduler of lr
scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, last_epoch=ini_epoch)  # 学习率衰退

# Create tensorboard SummaryWriter to monitor training
summary_writer = SummaryWriter(results_save_dir)

# Define loss
loss = nn.MSELoss(reduction='sum')

#############################################################################
## Define train, valid, and test functions
#############################################################################

y_mean = y_mean.to(device)
y_std = y_std.to(device)


def train(leave=False):
    model.train()
    sum_loss = 0

    total = len(train_dataset) / args.batch_size
    train_iter = enumerate(train_loader)
    t_train = tqdm(total=total, leave=leave)

    for i, data in train_iter:
        # Clear the cache
        optimizer.zero_grad()
        data = data.to(device)

        batch_out = model(data)
        batch_loss = loss(batch_out, data.y)

        batch_loss.backward()
        optimizer.step()

        # Calculate the reverse normalized loss
        with torch.no_grad():
            batch_out_reverse = utils.reverse_zscore(batch_out, y_mean, y_std)
            real_out_reverse = utils.reverse_zscore(data.y, y_mean, y_std)
            batch_loss_reverse = loss(batch_out_reverse, real_out_reverse)
            batch_loss_item = batch_loss_reverse.item()
            sum_loss += batch_loss_item

        # Output train loss, where loss denotes the average MSE produced by individual node in position
        t_train.set_description("train loss = %.5f" % (batch_loss_item / data.num_real_nodes.sum().item()),
                                refresh=False)
        t_train.update(1)

    return (sum_loss / train_dataset.num_real_nodes.sum().item())  # return loss of each epoch


def valid(leave=False):
    model.eval()
    sum_loss = 0

    total = len(val_dataset) / args.batch_size
    val_iter = enumerate(val_loader)
    t_val = tqdm(total=total, leave=leave)

    with torch.no_grad():
        for i, data in val_iter:
            data = data.to(device)
            batch_out = model(data)

            batch_out_reverse = utils.reverse_zscore(batch_out, y_mean, y_std)
            real_out_reverse = utils.reverse_zscore(data.y, y_mean, y_std)

            batch_loss = loss(batch_out_reverse, real_out_reverse)
            batch_loss_item = batch_loss.item()
            sum_loss += batch_loss_item
            t_val.set_description("valid loss = %.5f" % (batch_loss_item / data.num_real_nodes.sum().item()),
                                  refresh=False)
            t_val.update(1)
    return (sum_loss / val_dataset.num_real_nodes.sum().item())


def test(model, leave=False):
    model.eval()
    sum_loss = 0

    total = len(test_dataset) / args.batch_size
    t_test = tqdm(total=total, leave=leave)

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            batch_out = model(data)

            batch_out_reverse = utils.reverse_zscore(batch_out, y_mean, y_std)
            real_out_reverse = utils.reverse_zscore(data.y, y_mean, y_std)

            batch_loss = loss(batch_out_reverse, real_out_reverse)
            batch_loss_item = batch_loss.item()
            sum_loss += batch_loss_item
            t_test.set_description("test loss = %.5f" % (batch_loss_item / data.num_real_nodes.sum().item()),
                                   refresh=False)
            t_test.update(1)
    return (sum_loss / test_dataset.num_real_nodes.sum().item())


def model_train():
    min_valid_loss = 1e7
    train_loss = np.zeros(args.n_epochs)
    valid_loss = np.zeros(args.n_epochs)

    t = tqdm(range(ini_epoch, args.n_epochs))
    for epoch in t:
        train_loss[epoch] = train(leave=bool(epoch == args.n_epochs - 1 - ini_epoch))
        valid_loss[epoch] = valid(leave=bool(epoch == args.n_epochs - 1 - ini_epoch))
        scheduler.step()
        # write loss values to tensorboard summary_writer
        summary_writer.add_scalar('train_loss', train_loss[epoch], epoch)
        summary_writer.add_scalar('valid_loss', valid_loss[epoch], epoch)

        # Select the model with the lowest valid loss as the optimal model
        if valid_loss[epoch] < min_valid_loss:
            min_valid_loss = valid_loss[epoch].copy()
            min_train_loss = train_loss[epoch].copy()
            best_model = copy.deepcopy(model)

            # Save the best model
            torch.save({"model_state_dict": best_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "lr": optimizer.state_dict()['param_groups'][0]['lr']}, best_model_path)

        print('Epoch: {:02d}, Training Loss:   {:.4f}'.format(epoch, train_loss[epoch]))
        print('           Validation Loss: {:.4f}'.format(valid_loss[epoch]))

        # Save the model for the current epoch
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "lr": optimizer.state_dict()['param_groups'][0]['lr']}, model_path)

    # Record minimum train_loss and minimum val_loss
    summary_writer.add_hparams(config_dict,
                               {'min_train_loss': min_train_loss,
                                'min_valid_loss': min_valid_loss})


def model_evaluate(record=True):
    # Load model parameters
    model_path = f'{results_save_dir}/best_model.pth'
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device))['model_state_dict'])
    test_loss = test(model)
    print('Test Loss: {:.4f}'.format(test_loss))
    if record == True:
        summary_writer.add_hparams(config_dict,
                                   {'test_loss': test_loss})


#############################################################################
## Define the main function
#############################################################################

def main():
    if args.mode == 'train':
        model_train()
    elif args.mode == 'evaluate':
        model_evaluate()


if __name__ == '__main__':
    main()
