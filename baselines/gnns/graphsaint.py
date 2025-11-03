# -*- coding: utf-8 -*-
"""
GraphSAINT baseline for WikiArt dataset.

This implementation uses GraphSAINT random walk sampling with SAGEConv layers
for node classification, regression, or retrieval tasks on the WikiArt graph.
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

from sklearn.metrics import average_precision_score, classification_report
from torch_geometric.data import GraphSAINTRandomWalkSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import degree
from torch_geometric.data import Data


parser = argparse.ArgumentParser(description='GraphSAINT baseline for WikiArt')
parser.add_argument('--data_dir', type=str, default='graph_data/',
                    help='Root directory for data.')
parser.add_argument('--task', type=str, default='styles',
                    help='Task labels file name.')
parser.add_argument('--save_path', type=str, default=None,
                    help='Model save path. If None, model will not be saved.')
parser.add_argument('--sampler_save_dir', type=str, default='graphsaint_sampler/',
                    help='Directory to save GraphSAINT sampler cache.')
parser.add_argument('--task_type', type=str, default='classification',
                    choices=['classification', 'regression', 'retrieval'],
                    help='Type of task.')
parser.add_argument('--use_normalization', action='store_true',
                    help='Use edge normalization in GraphSAINT.')
parser.add_argument('--load_path', type=str, default=None,
                    help='Path to load trained model.')
parser.add_argument('--hidden_dim', type=int, default=256,
                    help='Hidden dimension for SAGEConv layers.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of training epochs.')
parser.add_argument('--walk_length', type=int, default=2,
                    help='Length of random walk.')
parser.add_argument('--num_steps', type=int, default=5,
                    help='Number of sampling steps per epoch.')
parser.add_argument('--sample_coverage', type=int, default=100,
                    help='Number of samples per node.')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='Batch size.')
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

# Print task information
print('='*80)
print(f'Task: {args.task}')
print(f'Task type: {args.task_type}')
print('='*80)

# Set save path if provided
# Set save path if provided
if args.save_path:
    args.save_path = 'saved_models/' + args.save_path + '_' + args.task + '_' + time
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

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

# Create PyTorch Geometric data object
edge_index = adj.coalesce().indices().cpu()
data = Data(edge_index=edge_index,
            test_mask=test_labels_mask.cpu(),
            train_mask=train_labels_mask.cpu(),
            val_mask=val_labels_mask.cpu(),
            x=features.cpu(),
            y=labels.cpu())
row, col = data.edge_index
data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Normalize by in-degree

# Create GraphSAINT sampler
print(f'[{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Creating GraphSAINT sampler...')
# Create sampler save directory if it doesn't exist and a path is provided
if args.sampler_save_dir and not os.path.exists(args.sampler_save_dir):
    os.makedirs(args.sampler_save_dir)

g = torch.Generator()
g.manual_seed(args.seed)

loader = GraphSAINTRandomWalkSampler(data, batch_size=args.batch_size,
                                     walk_length=args.walk_length,
                                     num_steps=args.num_steps,
                                     sample_coverage=args.sample_coverage,
                                     save_dir=args.sampler_save_dir,
                                     num_workers=args.num_workers,
                                     generator=g)


class Net(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Net, self).__init__()
        
        if args.task_type == 'regression':
            out_channels = 1
        elif args.task_type == 'retrieval':
            out_channels = args.retrieval_num_classes
            
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(3 * hidden_channels, out_channels)

    def forward(self, x0, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(x0, edge_index))
        x1 = F.dropout(x1, p=args.dropout, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, p=args.dropout, training=self.training)
        x3 = F.relu(self.conv3(x2, edge_index))
        x3 = F.dropout(x3, p=args.dropout, training=self.training)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.lin(x)
        
        if args.task_type == 'classification':
            return x.log_softmax(dim=-1)
        else:
            return x


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
    model.to(device)
    model.train()

    total_loss = total_examples = 0
    
    for data_batch in loader:
        data_batch = data_batch.to(device)
        optimizer.zero_grad()

        if args.use_normalization:
            out = model(data_batch.x, data_batch.edge_index)
            if args.task_type == 'classification':
                loss = F.nll_loss(out, data_batch.y, reduction='none')
                loss = (loss * data_batch.node_norm)[data_batch.train_mask].sum()
            elif args.task_type == 'regression':
                loss = nn.L1Loss()(out[data_batch.train_mask].flatten(),
                                   data_batch.y[data_batch.train_mask].float())
            elif args.task_type == 'retrieval':
                loss = nn.BCEWithLogitsLoss()(out[data_batch.train_mask],
                                              data_batch.y[data_batch.train_mask])
        else:
            out = model(data_batch.x, data_batch.edge_index)
            
            if args.task_type == 'classification':
                loss = F.nll_loss(out[data_batch.train_mask],
                                  data_batch.y[data_batch.train_mask])
            elif args.task_type == 'regression':
                loss = nn.L1Loss()(out[data_batch.train_mask].flatten(),
                                   data_batch.y[data_batch.train_mask].float())
            elif args.task_type == 'retrieval':
                loss = nn.BCEWithLogitsLoss()(out[data_batch.train_mask],
                                              data_batch.y[data_batch.train_mask])

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data_batch.num_nodes
        total_examples += data_batch.num_nodes
        
    return total_loss / total_examples


@torch.no_grad()
def test():
    """
    Evaluate the model on train/val/test sets.
    
    Returns:
        tuple: (losses, metrics) for train/val/test splits
            For classification/regression: metrics are accuracies
            For retrieval: metrics are mAP scores
    """
    model.to('cpu')
    model.eval()
    
    out = model(data.x.to('cpu'), data.edge_index.to('cpu'))
    pred = out.argmax(dim=-1)
   
    if args.task_type == 'classification':
        correct = pred.eq(data.y.to('cpu'))
    elif args.task_type == 'regression':
        correct = torch.abs(out.flatten() - data.y.to('cpu').float())
        
    losses = []
    metrics = []
    
    if args.task_type != 'retrieval':
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            if args.task_type == 'classification':
                losses.append(F.nll_loss(out[mask],
                                         data.y[mask].to('cpu')).item())
                metrics.append(correct[mask].sum().item() / mask.sum().item())
            elif args.task_type == 'regression':
                losses.append(nn.L1Loss()(out[mask].flatten(),
                                          data.y[mask].to('cpu').float()).item())
                within_threshold = sum(correct[mask] <= args.regression_threshold).item()
                metrics.append(within_threshold / mask.sum().item())
    else:
        for mask in [idx_train, idx_val, idx_test]:
            losses.append(nn.BCEWithLogitsLoss()(out[mask],
                                                 data.y[mask].to('cpu')).item())

            outputs_ = out[mask].sigmoid().cpu().detach().numpy()
            targets_ = data.y[mask].to('cpu')
            
            mAP = average_precision_score(targets_, outputs_, average='macro')
            metrics.append(mAP)
            
    return losses, metrics


# Initialize early stopping only if save_path is provided
best_model_state = None  # Store best model state in memory
if args.save_path:
    early_stopping = utils.EarlyStopping(path=args.save_path,
                                         patience=50, verbose=True)

# Determine metric name for printing
metric_name = 'Acc' if args.task_type in ['classification', 'regression'] else 'mAP'

# Training loop
print(f'[{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Starting training...')
best_val_loss = float('inf')

for epoch in range(1, args.epochs + 1):
    loss = train()

    if epoch % args.monitor_at == 0:
        losses, metrics = test()
        print(f'[{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
              f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Train {metric_name}: {metrics[0]:.4f}, Val {metric_name}: {metrics[1]:.4f}')
        
        # Save best model state in memory
        if losses[1] < best_val_loss:
            best_val_loss = losses[1]
            best_model_state = model.state_dict().copy()
        
        if args.save_path:
            early_stopping(epoch=epoch, val_loss=losses[1],
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

losses, metrics = test()
print(f'[{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
      f'Test {metric_name}: {metrics[2]:.4f}')