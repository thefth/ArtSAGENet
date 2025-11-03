# -*- coding: utf-8 -*-
"""
Training script for ArtSAGENet.

Trains ArtSAGENet models for single-task or multi-task artwork classification/regression.
"""
import argparse
import datetime
import helpers
import pickle
import torch
import torch.nn as nn
import utils

from artsagenet import ArtSAGENet
from loader import NeighborSamplerImages
from torchvision import transforms


parser = argparse.ArgumentParser(description='Train ArtSAGENet on WikiArt dataset')
parser.add_argument('--no-cuda', action='store_true',
                    help='Disable CUDA training.')
parser.add_argument('--data_dir', type=str, default='graph_data/',
                    help='Root directory for data.')
parser.add_argument('--adjacency', type=str, default='adj_artists_schools_random_128',
                    help='Adjacency matrix filename.')
parser.add_argument('--features', type=str, default='visual_features_resnet_34',
                    help='Visual features filename.')
parser.add_argument('--images', type=str, default='images.pkl',
                    help='Image paths pickle file.')
parser.add_argument('--labels_task1', type=str, default='labels_styles',
                    help='Labels for task 1.')
parser.add_argument('--labels_task2', type=str, default='labels_artists',
                    help='Labels for task 2.')
parser.add_argument('--labels_task3', type=str, default='labels_timelines',
                    help='Labels for task 3.')
parser.add_argument('--indices', type=str, default='indices.pkl',
                    help='Train/val/test split indices.')
parser.add_argument('--task_names', type=str, nargs=3, default=['Style', 'Artist', 'Timeline'],
                    help='Names for the three tasks (for multi-task).')
parser.add_argument('--save_path', type=str, default=None,
                    help='Model save path. If None, model will not be saved.')
parser.add_argument('--task_type', type=str, default='classification',
                    choices=['classification', 'regression', 'retrieval'],
                    help='Type of task for task 3 (task 1 and 2 are always classification).')
parser.add_argument('--no-fine_tune', action='store_true',
                    help='Disable fine-tuning of ResNet152.')
parser.add_argument('--no-multi_task', action='store_true',
                    help='Disable multi-task learning (single task only).')
parser.add_argument('--merge', type=str, default='concatenate',
                    choices=['concatenate', 'add', 'multiply', 'mean'],
                    help='Strategy for merging CNN and GNN features.')
parser.add_argument('--load_path', type=str, default=None,
                    help='Path to load pre-trained model.')
parser.add_argument('--hidden_dim', type=int, default=2048,
                    help='Hidden dimension for SAGEConv layers.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate.')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum.')
parser.add_argument('--scheduler_type',  type=str, default='plateau',
                    choices=['plateau', 'multistep'], 
                    help='Scheduler type.')                    
parser.add_argument('--num_k1', type=int, default=25,
                    help='Number of neighbors to sample at first hop.')
parser.add_argument('--num_k2', type=int, default=10,
                    help='Number of neighbors to sample at second hop.')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size.')
parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of data loading workers.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility.')
parser.add_argument('--regression_threshold', type=float, default=0.03355705,
                    help='Threshold for regression accuracy calculation.')

# Parse arguments
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.fine_tune = not args.no_fine_tune
args.multi_task = not args.no_multi_task

print(args)

# Set random seed
utils.set_seed(args.seed)

# Set save path if provided
if args.save_path:
    args.save_path = args.save_path + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Load data
print(f'[{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Loading data...')
adj = torch.load(args.data_dir + '/' + args.adjacency, map_location='cpu')
features = torch.load(args.data_dir + '/' + args.features).to('cuda')
images = pickle.load(open(args.data_dir + '/' + args.images, 'rb'))

if args.multi_task:
    task1_labels = torch.load(args.data_dir + '/' + args.labels_task1).to('cuda')
    task2_labels = torch.load(args.data_dir + '/' + args.labels_task2).to('cuda')
    task3_labels = torch.load(args.data_dir + '/' + args.labels_task3).to('cuda')
    labels = [task1_labels, task2_labels, task3_labels]
else:
    labels = torch.load(args.data_dir + '/' + args.labels_task1).to('cuda')

indices = pickle.load(open(args.data_dir + '/' + args.indices, 'rb'))
idx_train = torch.tensor(indices['train'], dtype=torch.int)
idx_val = torch.tensor(indices['val'], dtype=torch.int)
idx_test = torch.tensor(indices['test'], dtype=torch.int)

# Define image transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Create a generator with fixed seed for DataLoaders
g = torch.Generator()
g.manual_seed(args.seed)

# Create data loaders
print(f'[{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Creating data loaders...')
train_loader = NeighborSamplerImages(images, data_transforms['train'],
                                     adj.coalesce().indices().cpu(),
                                     node_idx=idx_train,
                                     sizes=[args.num_k1, args.num_k2],
                                     batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers,
                                     generator=g)

val_loader = NeighborSamplerImages(images, data_transforms['val'], 
                                   adj.coalesce().indices().cpu(),
                                   node_idx=idx_val,
                                   sizes=[args.num_k1, args.num_k2],
                                   batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers,
                                   generator=g)

test_loader = NeighborSamplerImages(images, data_transforms['test'], 
                                    adj.coalesce().indices().cpu(),
                                    node_idx=idx_test,
                                    sizes=[args.num_k1, args.num_k2],
                                    batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers,
                                    generator=g)

dataloader_dict = {'train': train_loader, 'val': val_loader, 'test': test_loader}
dataset_sizes = {'train': len(idx_train), 'val': len(idx_val), 'test': len(idx_test)}

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Using device: {device}')

# Create model
print(f'[{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Creating model...')
if args.multi_task:
    # Get number of classes for each task
    out_dim_task1 = labels[0].max().item() + 1  # Single-label
    out_dim_task2 = labels[1].max().item() + 1  # Single-label
    
    if args.task_type == 'regression':
        out_dim_task3 = 1
    else:  # classification
        out_dim_task3 = labels[2].max().item() + 1  # Single-label
    
    model = ArtSAGENet(features.shape[1], args.hidden_dim,
                       [out_dim_task1, out_dim_task2, out_dim_task3],
                       dropout=args.dropout, fine_tune=args.fine_tune, 
                       merge=args.merge, multitask=args.multi_task)
else:
    if args.task_type == 'regression':
        out_dim = 1
    elif args.task_type == 'retrieval':
        out_dim = labels.shape[1]  # Multi-label (one-hot)
    else:  # classification
        out_dim = labels[0].max().item() + 1  # Single-label
    
    model = ArtSAGENet(features.shape[1], args.hidden_dim, out_dim,
                       dropout=args.dropout, fine_tune=args.fine_tune, 
                       merge=args.merge, multitask=args.multi_task)

model = model.to(device)

# Configure trainable parameters
if args.fine_tune:
    # Fine-tune only layer4 and fc of ResNet152
    for param in model.cnn.parameters():
        param.requires_grad = False
        
    for param in model.cnn.layer4.parameters():
        param.requires_grad = True
    
    for param in model.cnn.fc.parameters():
        param.requires_grad = True
        
    params = (list(model.cnn.layer4.parameters()) +
              list(model.cnn.fc.parameters()) +
              list(model.gnn.parameters()) +
              list(model.task1.parameters()))
    
    if args.multi_task:
        params += list(model.task2.parameters()) + list(model.task3.parameters())
else:
    params = list(model.parameters())

# Define loss functions
if args.multi_task:
    if args.task_type == 'classification':
        criterion = [nn.CrossEntropyLoss(),
                     nn.CrossEntropyLoss(),
                     nn.CrossEntropyLoss()]
    else:  # regression for task 3
        criterion = [nn.CrossEntropyLoss(),
                     nn.CrossEntropyLoss(),
                     nn.L1Loss()]
else:
    if args.task_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    elif args.task_type == 'regression':
        criterion = nn.L1Loss()
    elif args.task_type == 'retrieval': 
        criterion = nn.BCEWithLogitsLoss()

# Define optimizer and scheduler
optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)

if args.scheduler_type == 'plateau':
    exp_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, verbose=True)
elif args.scheduler_type == 'multistep':
    exp_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[15, 30, 40], gamma=0.1, verbose=True)
        
# Print training configuration
print('=' * 80)
if args.multi_task:
    print(f'Training mode: Multi-task learning')
    print(f'Tasks: {args.task_names[0]}, {args.task_names[1]}, {args.task_names[2]}')
else:
    print(f'Training mode: Single-task learning')
    print(f'Task: {args.task_names[0]}')
print(f'Task type: {args.task_type}')
print(f'Merging operation: {args.merge}')
print('=' * 80)
        
# Print model architecture
print(f'[{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Model architecture:')
print(model)
print('='*80)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total parameters: {total_params:,}')
print(f'Trainable parameters: {trainable_params:,}')
print('='*80)

# Train model
if args.multi_task:
    model, epoch_loss, epoch = helpers.train_model_multitask(model, dataloader_dict,
                                          features, labels, dataset_sizes,
                                          criterion, optimizer, exp_lr_scheduler,
                                          task_type=args.task_type,
                                          save_path=args.save_path,
                                          regression_threshold=args.regression_threshold,
                                          task_names=args.task_names,
                                          device=device, num_epochs=args.epochs)
else:
    model, epoch_loss, epoch = helpers.train_model(model, dataloader_dict,
                                features, labels, dataset_sizes,
                                criterion, optimizer, exp_lr_scheduler,
                                task_type=args.task_type,
                                save_path=args.save_path,
                                regression_threshold=args.regression_threshold,
                                device=device, num_epochs=args.epochs)

# Evaluate on test set
print("="*80)
if args.multi_task:
    helpers.test_model_multitask(model, dataloader_dict,
                                 features, labels, dataset_sizes,
                                 criterion, task_type=args.task_type,
                                 regression_threshold=args.regression_threshold,
                                 task_names=args.task_names,
                                 device=device)
else:
    helpers.test_model(model, dataloader_dict,
                      features, labels, dataset_sizes,
                      criterion, task_type=args.task_type,
                      regression_threshold=args.regression_threshold,
                      device=device)