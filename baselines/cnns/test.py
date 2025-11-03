# -*- coding: utf-8 -*-
"""
Testing script for CNN baselines.

This script evaluates trained CNN models on the WikiArt test set.
"""
import torch
import os
import datetime
import pickle
import argparse
import utils
import helpers


parser = argparse.ArgumentParser(description='Test CNN baselines on WikiArt dataset')
parser.add_argument('--no-cuda', action='store_true',
                    help='Disables CUDA training.')
parser.add_argument('--data_path', type=str, default='./dataset.pkl',
                    help='Path to dataset pickle file.')
parser.add_argument('--image_dir', type=str, default='WikiArt/Dataset/',
                    help='Directory containing images.')
parser.add_argument('--model', type=str, default='resnet-152',
                    choices=['alexnet', 'vgg-16bn', 'vgg-19bn', 'resnet-34', 'resnet-152'],
                    help='Model architecture to use.')
parser.add_argument('--no-fine_tune', action='store_true',
                    help='Model was trained from scratch (not fine-tuned).')
parser.add_argument('--single_task', type=int, default=1,
                    help='Task index for single-task learning (1, 2, or 3).')
parser.add_argument('--no-multi_task', action='store_true',
                    help='Use single-task model (not multi-task).')
parser.add_argument('--mask', type=int, default=-100,
                    help='Mask value for labels.')
parser.add_argument('--task_type', type=str, default='classification',
                    choices=['classification', 'regression', 'retrieval'],
                    help='Type of task.')
parser.add_argument('--load_path', type=str, default='model.tar',
                    help='Path to model checkpoint to load.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate (used for optimizer initialization).')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Momentum for SGD optimizer.')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size for testing.')
parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of workers for data loading.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.fine_tune = not args.no_fine_tune
args.multi_task = not args.no_multi_task
    
# Save arguments to file
time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

print(args)

# Set random seed for reproducibility
utils.set_seed(args.seed)

data_path = args.data_path

print('Loading data from folder: ' + data_path[:data_path.index('/')])

data = pickle.load(open(data_path, 'rb'))

if args.cuda:
    device = utils.set_gpu()
else:
    device = torch.device('cpu')

# Create datasets and dataloaders
dataloader, dataset_size = helpers.create_dataset(
    data,
    image_dir=args.image_dir,
    batch_size=args.batch_size,
    task=args.single_task,
    multitask=args.multi_task
)
                                                          
# Initialize model, criterion, optimizer, and scheduler
model_conv, criterion, optimizer_conv, exp_lr_scheduler = helpers.get_model(
    model=args.model, 
    fine_tune=args.fine_tune, 
    device=device, 
    num_classes_task1=dataloader['train'].dataset[0][-1][0].shape[0] 
                      if args.multi_task 
                      else dataloader['train'].dataset[0][-1][0].shape[0],
    num_classes_task2=dataloader['train'].dataset[0][-1][1].shape[0] 
                      if args.multi_task and len(dataloader['train'].dataset[0][-1]) > 1 
                      else None,
    num_classes_task3=dataloader['train'].dataset[0][-1][-1].shape[0] 
                      if len(list(dataloader['train'].dataset[0][-1][-1].size())) > 0 
                      else 1,
    multitask=args.multi_task,
    task_type=args.task_type, 
    lr=args.lr, 
    momentum=args.momentum
)
                                                                                    
# Load trained model checkpoint
model_conv, optimizer_conv, exp_lr_scheduler, epoch, val_loss = utils.load_model(
    model_conv, 
    optimizer_conv, 
    exp_lr_scheduler,
    args.load_path
)

# Test model
if not args.multi_task:
    model, loss, acc = helpers.test_model(
        model_conv, dataloader, 
        dataset_size, criterion,
        optimizer_conv,
        task_type=args.task_type,
        device=device
    )
else:
    model, loss, acc_1, acc_2, acc_3 = helpers.test_model_multitask(
        model_conv,
        dataloader, 
        dataset_size,
        criterion, 
        optimizer_conv, 
        task_type=args.task_type, 
        device=device
    )