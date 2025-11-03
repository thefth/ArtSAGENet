# -*- coding: utf-8 -*-
"""
Training script for CNN baselines.

This script trains CNN models (AlexNet, VGG, ResNet) on the WikiArt dataset
for single-task or multi-task learning.
"""
import torch
import os
import datetime
import pickle
import argparse
import utils
import helpers


parser = argparse.ArgumentParser(description='Train CNN baselines on WikiArt dataset')
parser.add_argument('--no-cuda', action='store_true',
                    help='Disables CUDA training.')
parser.add_argument('--data_path', type=str, default='./dataset.pkl',
                    help='Path to dataset pickle file.')
parser.add_argument('--image_dir', type=str, default='WikiArt/Dataset/',
                    help='Directory containing images.')
parser.add_argument('--model', type=str, default='resnet-152',
                    choices=['alexnet', 'vgg-16bn', 'vgg-19bn', 'resnet-34', 'resnet-152'],
                    help='Model architecture to use.')
parser.add_argument('--save_path', type=str, default=None,
                    help='Model save path.')
parser.add_argument('--no-fine_tune', action='store_true',
                    help='Disables fine-tuning (train from scratch).')
parser.add_argument('--single_task', type=int, default=None, choices=[1, 2, 3],
                    help='Task index for single-task learning (1, 2, or 3). If None, multi-task learning is used.')
parser.add_argument('--mask', type=int, default=-100,
                    help='Mask value for labels.')
parser.add_argument('--task_type', type=str, default='classification',
                    choices=['classification', 'regression', 'retrieval'],
                    help='Type of task.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate.')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Momentum for SGD optimizer.')
parser.add_argument('--scheduler_type',  type=str, default='plateau',
                    choices=['plateau', 'multistep'], 
                    help='Scheduler type.')    
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs.')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size for training.')
parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of workers for data loading.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility.')
parser.add_argument('--regression_threshold', type=float, default=0.03355705,
                    help='Threshold for regression accuracy calculation.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.fine_tune = not args.no_fine_tune
args.multi_task = (args.single_task is None)

# Create directory structure for saving results
path = args.model
    
# Save arguments to file
time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

print(args)

# Set random seed for reproducibility
utils.set_seed(args.seed)

if args.save_path:
    args.save_path = os.path.join(path, args.save_path + time)

data_path = args.data_path

print('Loading data from folder: ' + data_path[:data_path.index('/')])

data = pickle.load(open(data_path, 'rb'))

if args.cuda:
    device = utils.set_gpu()
    
# Create datasets and dataloaders
dataloader, dataset_size = helpers.create_dataset(data,
                                                 image_dir=args.image_dir,
                                                 batch_size=args.batch_size,
                                                 task=args.single_task,
                                                 multitask=args.multi_task)
    
# Initialize model, criterion, optimizer, and scheduler
if args.multi_task:
    # Multi-task: provide all three task dimensions
    model_conv, criterion, optimizer_conv, exp_lr_scheduler = helpers.get_model(
        model=args.model, 
        fine_tune=args.fine_tune,
        multitask=args.multi_task,
        mask=args.mask, 
        num_classes_task1=dataloader['train'].dataset[0][1][0].shape[0],
        num_classes_task2=dataloader['train'].dataset[0][1][1].shape[0], 
        num_classes_task3=dataloader['train'].dataset[0][1][-1].shape[0] 
                          if args.task_type != 'regression' else 1,
        scheduler_type=args.scheduler_type,
        device=device, 
        task_type=args.task_type, 
        lr=args.lr, 
        momentum=args.momentum,
        monitor_accuracy=False
    )
else:
    # Single-task: only provide the dimension for the single task
    if args.task_type == 'regression':
        num_classes = 1
    else:
        # For classification or retrieval, get shape from the label tensor
        num_classes = dataloader['train'].dataset[0][1][0].shape[0]
    
    model_conv, criterion, optimizer_conv, exp_lr_scheduler = helpers.get_model(
        model=args.model, 
        fine_tune=args.fine_tune,
        multitask=args.multi_task,
        mask=args.mask, 
        num_classes_task1=num_classes,
        num_classes_task2=None, 
        num_classes_task3=None,
        scheduler_type=args.scheduler_type,
        device=device, 
        task_type=args.task_type, 
        lr=args.lr, 
        momentum=args.momentum,
        monitor_accuracy=False
    )
        
utils.print_model(model_conv, dataloader, device=device, verbose=True)

# Train model
if not args.multi_task:
    model, loss, epoch = helpers.train_model(
        model_conv, dataloader, 
        dataset_size, criterion,
        optimizer_conv,
        exp_lr_scheduler,
        task_type=args.task_type,
        monitor_accuracy=False,
        save_path=args.save_path,
        device=device,
        num_epochs=args.epochs
    )
else:
    model, loss, epoch = helpers.train_model_multitask(
        model_conv, dataloader, 
        dataset_size, criterion,
        optimizer_conv,
        exp_lr_scheduler,
        task_type=args.task_type,
        monitor_accuracy=False,
        save_path=args.save_path,
        device=device,
        num_epochs=args.epochs
    )
    
# Evaluate on test set
print("="*80)
if args.multi_task:
    helpers.test_model_multitask(model, dataloader,
                                 dataset_size, criterion, optimizer_conv,
                                 task_type=args.task_type, 
                                 regression_threshold=args.regression_threshold,
                                 device=device)
else:
    helpers.test_model(model, dataloader,
                       dataset_size, criterion, optimizer_conv,
                       task_type=args.task_type, 
                       regression_threshold=args.regression_threshold,
                       device=device)