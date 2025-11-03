# -*- coding: utf-8 -*-
"""
GraphSAGE baseline for WikiArt dataset.

This implementation uses GraphSAGE with neighbor sampling for
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

from sklearn.metrics import average_precision_score, classification_report
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data


parser = argparse.ArgumentParser(description='GraphSAGE baseline for WikiArt')
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
                    help='Hidden dimension for SAGEConv layers.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate.')
parser.add_argument('--num_k1', type=int, default=25,
                    help='Number of neighbors to sample at first hop.')
parser.add_argument('--num_k2', type=int, default=10,
                    help='Number of neighbors to sample at second hop.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of training epochs.')
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
data = Data(edge_index=adj.coalesce().indices(), test_mask=test_labels_mask,
            train_mask=train_labels_mask, val_mask=val_labels_mask,
            x=features, y=labels)
            
g = torch.Generator()
g.manual_seed(args.seed)

# Create neighbor samplers for training and inference
print(f'[{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Creating neighbor samplers...')
train_loader = NeighborSampler(data.edge_index, node_idx=idx_train,
                               sizes=[args.num_k1, args.num_k2],
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers,
                               generator=g)
subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers,
                                  generator=g)


class SAGE(nn.Module):
    """
    GraphSAGE model with 2-layer architecture.
    
    Supports classification, regression, and retrieval tasks.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        Args:
            in_channels (int): Number of input features
            hidden_channels (int): Number of hidden features
            out_channels (int): Number of output classes/features
        """
        super(SAGE, self).__init__()

        self.num_layers = 2
        
        if args.task_type == 'regression':
            out_channels = 1
        elif args.task_type == 'retrieval':
            out_channels = args.retrieval_num_classes

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, adjs):
        """
        Forward pass using sampled neighborhoods.
        
        Args:
            x (torch.Tensor): Node features
            adjs (list): List of (edge_index, e_id, size) tuples from sampler
            
        Returns:
            torch.Tensor: Output predictions
        """
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=args.dropout, training=self.training)
                
        if args.task_type == 'classification':
            return x.log_softmax(dim=-1)
        else:
            return x
            
    def inference(self, x_all):
        """
        Inference on the full graph using mini-batch processing.
        
        Args:
            x_all (torch.Tensor): All node features
            
        Returns:
            torch.Tensor: Node representations
        """
        # Move model to CPU for full-graph inference
        self.to('cpu')
        x_all = x_all.cpu()
        
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj  # Don't move to device - keep on CPU
                x = x_all[n_id]  # Already on CPU
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x)  # Already on CPU

            x_all = torch.cat(xs, dim=0)

        return x_all


# Initialize device, model and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Using device: {device}')

model = SAGE(data.x.shape[1], args.hidden_dim,
             data.y.max().item() + 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Send features and labels to device
x = data.x.to(device)
y = data.y.squeeze().to(device)

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

    total_loss = 0
    
    for batch_size, n_id, adjs in train_loader:
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        
        if args.task_type == 'classification':
            loss = F.nll_loss(out, y[n_id[:batch_size]])
        elif args.task_type == 'regression':
            loss = nn.L1Loss()(out.flatten(),
                               y[n_id[:batch_size]].float())
        elif args.task_type == 'retrieval':
            loss = nn.BCEWithLogitsLoss()(out, y[n_id[:batch_size]])
            
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
    
    return total_loss / len(train_loader)


@torch.no_grad()
def test():
    """
    Evaluate the model on train/val/test sets.
    
    Returns:
        tuple: (losses, metrics) for train/val/test splits
            For classification/regression: metrics are accuracies
            For retrieval: metrics are mAP scores
    """
    model.eval()
    y_true = y.cpu().unsqueeze(-1)
    
    if args.task_type == 'classification':
        out = model.inference(x)  # This moves model to CPU internally
        out = out.log_softmax(dim=-1)
        y_pred = out.argmax(dim=-1, keepdim=True)
    elif args.task_type == 'regression':
        out = model.inference(x)
        correct = torch.abs(out.flatten() - data.y.to('cpu').float())
    elif args.task_type == 'retrieval':
        out = model.inference(x)
        
    losses = []
    metrics = []
    
    if args.task_type != 'retrieval':
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            if args.task_type == 'classification':
                losses += [F.nll_loss(out[mask],
                                      y[mask].to('cpu')).item()]
                metrics += [int(y_pred[mask].eq(y_true[mask]).sum()) /
                            int(mask.sum())]
            elif args.task_type == 'regression':
                losses += [nn.L1Loss()(out[mask].flatten(),
                                       data.y[mask].to('cpu').float()).item()]
                within_threshold = sum(correct[mask] <= args.regression_threshold).item()
                metrics += [within_threshold / mask.sum().item()]
    else:
        for mask in [idx_train, idx_val, idx_test]:
            losses.append(nn.BCEWithLogitsLoss()(out[mask],
                                                 data.y[mask].to('cpu')).item())
            
            outputs_ = out[mask].sigmoid().cpu().detach().numpy()
            targets_ = data.y[mask].cpu().detach().numpy()
            
            mAP = average_precision_score(targets_, outputs_, average='macro')
            metrics.append(mAP)

    # Move model back to GPU for next training epoch
    model.to(device)
    
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