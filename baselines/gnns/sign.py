# -*- coding: utf-8 -*-
"""
SIGN (Scalable Inception Graph Neural Network) baseline for WikiArt dataset.

This implementation uses SIGN with precomputed k-hop neighbors for
node classification, regression, or retrieval tasks on the WikiArt graph.
"""
import argparse
import datetime
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import torch_geometric.transforms as T

from sklearn.metrics import average_precision_score, classification_report
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.utils import add_self_loops
from torch_sparse import SparseTensor, matmul


parser = argparse.ArgumentParser(description='SIGN baseline for WikiArt')
parser.add_argument('--data_dir', type=str, default='graph_data/',
                    help='Root directory for data.')
parser.add_argument('--task', type=str, default='styles',
                    help='Task labels file name.')
parser.add_argument('--save_path', type=str, default=None,
                    help='Model save path. If None, model will not be saved.')
parser.add_argument('--task_type', type=str, default='classification',
                    choices=['classification', 'regression', 'retrieval'],
                    help='Type of task.')
parser.add_argument('--load_path', type=str, default=None,
                    help='Path to load trained model.')
parser.add_argument('--hidden_dim', type=int, default=2048,
                    help='Hidden dimension for linear layers.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of training epochs.')
parser.add_argument('--K', type=int, default=3,
                    help='Number of hops for feature propagation.')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='Base batch size.')
parser.add_argument('--batch_multiplier', type=int, default=32,
                    help='Multiplier for actual batch size.')
parser.add_argument('--monitor_at', type=int, default=1,
                    help='Evaluate model every monitor_at epochs.')
parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of data loading workers.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility.')
parser.add_argument('--regression_threshold', type=float, default=0.03355705,
                    help='Threshold for regression accuracy calculation.')
parser.add_argument('--retrieval_num_classes', type=int, default=54,
                    help='Number of classes for retrieval task.')
parser.add_argument('--verbose', action='store_true',
                    help='Print detailed model architecture.')

# Parse arguments, set seed and starting time
args = parser.parse_args()
print(args)
time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
utils.set_seed(args.seed)

# Set save path if provided
if args.save_path:
    args.save_path = 'saved_models/' + args.save_path + '_' + args.task + '_' + time
    # Create directory if it doesn't exist
    save_dir = os.path.dirname(args.save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f'[{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Created directory: {save_dir}')

# Load adjacency, node features, labels and split indices
print(f'[{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Loading data...')
adj = torch.load(args.data_dir + '/adj_artists_schools_random_128',
                 map_location='cpu')
features = torch.load(args.data_dir + '/visual_features_resnet_34',
                      map_location='cpu')
labels = torch.load(args.data_dir + '/labels_' + args.task,
                    map_location='cpu')
indices = pickle.load(open(args.data_dir + '/indices.pkl', 'rb'))

idx_train = torch.tensor(indices['train'], dtype=torch.long)
idx_val = torch.tensor(indices['val'], dtype=torch.long)
idx_test = torch.tensor(indices['test'], dtype=torch.long)

train_labels_mask = torch.zeros(labels.shape, dtype=torch.bool)
train_labels_mask[idx_train] = True
val_labels_mask = torch.zeros(labels.shape, dtype=torch.bool)
val_labels_mask[idx_val] = True
test_labels_mask = torch.zeros(labels.shape, dtype=torch.bool)
test_labels_mask[idx_test] = True

# Precompute k-hop features manually
print(f'[{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Precomputing {args.K}-hop features with SIGN...')

# Normalize features (row-wise L1 normalization)
row_sum = features.sum(dim=-1, keepdim=True)
features_norm = features / (row_sum + 1e-12)

# Get edge index and add self-loops
edge_index = adj.coalesce().indices().cpu()
edge_index, _ = add_self_loops(edge_index, num_nodes=features.size(0))

# Create sparse adjacency matrix
num_nodes = features.size(0)
adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                     sparse_sizes=(num_nodes, num_nodes))

# Normalize adjacency: D^{-1/2} A D^{-1/2}
deg = adj_t.sum(dim=1).to(torch.float)
deg_inv_sqrt = deg.pow(-0.5)
deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

# Compute multi-hop features: X, AX, A^2X, ..., A^K X
xs = [features_norm.cpu()]
for k in range(1, args.K + 1):
    xs.append(matmul(adj_t, xs[-1]))

print(f'[{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Created {args.K + 1} feature matrices.')

# Create data object with all features
data = Data(y=labels.cpu(),
            train_mask=train_labels_mask.cpu(),
            val_mask=val_labels_mask.cpu(),
            test_mask=test_labels_mask.cpu())

data.x = xs[0]
for k in range(1, args.K + 1):
    setattr(data, f'x{k}', xs[k])

train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
val_idx = data.val_mask.nonzero(as_tuple=False).view(-1)
test_idx = data.test_mask.nonzero(as_tuple=False).view(-1)

# Create data loaders
train_loader = DataLoader(train_idx,
                          batch_size=args.batch_multiplier * args.batch_size,
                          shuffle=True)
val_loader = DataLoader(val_idx,
                        batch_size=args.batch_multiplier * args.batch_size * 2)
test_loader = DataLoader(test_idx,
                         batch_size=args.batch_multiplier * args.batch_size * 2)


class Net(nn.Module):
    """
    SIGN model with separate linear layers for each hop and concatenation.
    
    Supports classification, regression, and retrieval tasks.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        Args:
            in_channels (int): Number of input features
            hidden_channels (int): Number of hidden features
            out_channels (int): Number of output classes/features
        """
        super(Net, self).__init__()

        if args.task_type == 'regression':
            out_channels = 1
        elif args.task_type == 'retrieval':
            out_channels = args.retrieval_num_classes
            
        self.lins = nn.ModuleList()
        for _ in range(args.K + 1):
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            
        self.lin = nn.Linear((args.K + 1) * hidden_channels, out_channels)

    def forward(self, xs):
        """
        Forward pass through the network.
        
        Args:
            xs (list): List of node features at different hops
            
        Returns:
            torch.Tensor: Output predictions
        """
        hs = []
        for x, lin in zip(xs, self.lins):
            h = lin(x).relu()
            h = F.dropout(h, p=args.dropout, training=self.training)
            hs.append(h)
            
        h = torch.cat(hs, dim=-1)
        h = self.lin(h)
        
        if args.task_type == 'classification':
            return h.log_softmax(dim=-1)
        else:
            return h


# Initialize device, model and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Using device: {device}')

model = Net(data.x.shape[1], args.hidden_dim,
            data.y.max().item() + 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Print model architecture
print(f'[{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Model architecture:')
print(model)
print('='*80)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total parameters: {total_params:,}')
print(f'Trainable parameters: {trainable_params:,}')
print('='*80)


def train():
    """
    Train the model for one epoch.
    
    Returns:
        float: Average training loss
    """
    model.train()

    total_loss = total_examples = 0
    
    for idx in train_loader:
        xs = [data.x[idx].to(device)]
        xs += [data[f'x{i}'][idx].to(device) for i in range(1, args.K + 1)]
        y = data.y[idx].to(device)

        optimizer.zero_grad()
        out = model(xs)
        
        if args.task_type == 'classification':
            loss = F.nll_loss(out, y)
        elif args.task_type == 'regression':
            loss = nn.L1Loss()(out.flatten(), y.float())
        elif args.task_type == 'retrieval':
            loss = nn.BCEWithLogitsLoss()(out, y)
            
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * idx.numel()
        total_examples += idx.numel()

    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    """
    Evaluate the model on a given data loader.
    
    Args:
        loader (DataLoader): Data loader for evaluation
        
    Returns:
        tuple: (total_loss, metric)
            For classification/regression: metric is accuracy
            For retrieval: metric is mAP
    """
    model.eval()

    total_loss = total_correct = total_examples = 0
    
    if args.task_type == 'retrieval':
        outputs = []
        targets = []
    
    for idx in loader:
        xs = [data.x[idx].to(device)]
        xs += [data[f'x{i}'][idx].to(device) for i in range(1, args.K + 1)]
        y = data.y[idx].to(device)

        out = model(xs)
        
        if args.task_type == 'classification':
            loss = F.nll_loss(out, y)
            total_correct += int((out.argmax(dim=-1) == y).sum())
        elif args.task_type == 'regression':
            loss = nn.L1Loss()(out.flatten(), y.float())
            total_correct += sum(torch.abs(out.flatten() -
                                           y.float()) <= args.regression_threshold)
        elif args.task_type == 'retrieval':
            loss = nn.BCEWithLogitsLoss()(out, y)
            outputs.extend(out.sigmoid().cpu())
            targets.extend(y.cpu())
                        
        total_loss += float(loss) * idx.numel()
        total_examples += idx.numel()
        
    if args.task_type == 'retrieval':
        outputs_ = torch.stack(outputs)
        targets_ = torch.stack(targets)
        mAP = average_precision_score(targets_.detach().numpy(),
                                      outputs_.detach().numpy(),
                                      average='macro')
        return total_loss / total_examples, mAP
    else:
        return total_loss / total_examples, total_correct / total_examples


# Initialize early stopping only if save_path is provided
best_model_state = None  # Store best model state in memory
if args.save_path:
    early_stopping = utils.EarlyStopping(path=args.save_path, patience=50,
                                         verbose=True)

# Determine metric name for printing
metric_name = 'Acc' if args.task_type in ['classification', 'regression'] else 'mAP'

# Training loop
print(f'[{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Starting training...')
best_val_loss = float('inf')

for epoch in range(1, args.epochs + 1):
    loss = train()
    
    if epoch % args.monitor_at == 0:
        train_loss, train_metric = test(train_loader)
        val_loss, val_metric = test(val_loader)
        print(f'[{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
              f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Train {metric_name}: {train_metric:.4f}, Val {metric_name}: {val_metric:.4f}')
        
        # Save best model state in memory
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if args.save_path:
            early_stopping(epoch=epoch, val_loss=val_loss,
                           model=model, optimizer=optimizer)
        
            if early_stopping.early_stop:
                print(f'[{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
                      f'Early stopping')
                break
    else:
        print(f'[{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
              f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# Final evaluation on test set with best model
print(f'[{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Training complete. Evaluating on test set...')
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f'[{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Using best model based on validation loss.')

test_loss, test_metric = test(test_loader)
print(f'[{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
      f'Test {metric_name}: {test_metric:.4f}')