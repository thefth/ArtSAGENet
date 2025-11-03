# -*- coding: utf-8 -*-
"""
Utility functions for CNN baselines.

This module provides utility classes and functions including:
- Early stopping mechanism
- Multi-task learning block
- Seed setting for reproducibility
- GPU configuration
- TensorBoard setup
- Model loading utilities
"""
import numpy as np
import os
import random
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


class EarlyStopping:
    """
    Early stopping to stop training when validation performance stops improving.
    """
    def __init__(self, accuracy=False, patience=10,
                 verbose=False, delta=0, save_path='checkpoint.pt'):
        """
        Args:
            accuracy (bool): If True, monitor accuracy (higher is better).
                           If False, monitor loss (lower is better).
            patience (int): Number of epochs with no improvement before stopping.
                          Default: 10
            verbose (bool): If True, prints messages when saving checkpoints.
                          Default: False
            delta (float): Minimum change to qualify as an improvement.
                         Default: 0
            save_path (str): Path for saving the checkpoint.
                      Default: 'checkpoint.pt'        
        """
        self.accuracy = accuracy
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.save_path = save_path
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        

    def __call__(self, epoch, current_score, model, optimizer, lr_scheduler):
        """
        Check if training should stop based on current score.
        
        Args:
            epoch (int): Current epoch number
            current_score (float): Current validation score (loss or accuracy)
            model (nn.Module): Model to save if performance improves
            optimizer: Optimizer state to save
            lr_scheduler: Learning rate scheduler state to save
        """
        score = current_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, current_score, model, optimizer, lr_scheduler)
        elif (not self.accuracy and score >= self.best_score + self.delta) or \
             (self.accuracy and score <= self.best_score + self.delta):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch, current_score, model, optimizer, lr_scheduler)
            self.counter = 0

    def save_checkpoint(self, epoch, current_score, model, optimizer, lr_scheduler):
        """
        Save model checkpoint when validation performance improves.
        
        Args:
            epoch (int): Current epoch number
            current_score (float): Current validation score
            model (nn.Module): Model to save
            optimizer: Optimizer state to save
            lr_scheduler: Learning rate scheduler state to save
        """
        if self.verbose:
            print(f'Validation score improved ({self.val_loss_min:.6f} --> {current_score:.6f}). Saving model ...')
        
        self.val_loss_min = current_score
        
        if self.save_path:
            torch.save({
                'epoch': epoch,
                'current_score': current_score,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            }, self.save_path)
        

class Multitask_Block(nn.Module):
    """
    Multi-task learning header block with separate output layers for each task.
    """
    def __init__(self, num_in_features, num_classes_task1,
                 num_classes_task2, num_classes_task3):
        """
        Args:
            num_in_features (int): Number of input features to the block.
            num_classes_task1 (int): Number of output classes for task 1.
            num_classes_task2 (int): Number of output classes for task 2.
            num_classes_task3 (int): Number of output classes for task 3.
        """
        super(Multitask_Block, self).__init__()

        in_features = num_in_features
        self.task1 = nn.Linear(in_features, num_classes_task1)
        self.task2 = nn.Linear(in_features, num_classes_task2)
        self.task3 = nn.Linear(in_features, num_classes_task3)
        
    def forward(self, x):
        """
        Forward pass through all task-specific layers.
        
        Args:
            x (torch.Tensor): Input features
            
        Returns:
            tuple: (task1_output, task2_output, task3_output)
        """
        task1 = self.task1(x)
        task2 = self.task2(x)
        task3 = self.task3(x)
        
        return task1, task2, task3


def set_seed(seed=42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True, warn_only=True)
    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
   
def set_gpu():
    """
    Configure GPU device for training.
    
    Returns:
        torch.device: CUDA device if available, otherwise CPU
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Training on:', device)
    
    return device
   
 
def set_tensorboard(tensorboard_path):
    """
    Initialize TensorBoard writer.
    
    Args:
        tensorboard_path (str): Path to save TensorBoard logs
        
    Returns:
        SummaryWriter: TensorBoard writer instance
    """
    return SummaryWriter(tensorboard_path)


def print_model(model, dataloader, device=torch.device('cpu'), verbose=False):
    """
    Print model architecture and optionally detailed summary.
    
    Args:
        model (nn.Module): Model to print
        dataloader (dict): Dictionary of dataloaders
        device (torch.device): Device to use for summary
        verbose (bool): If True, print detailed model summary with layer shapes
    """
    print('Model architecture:')
    
    if verbose:
        summary(model.to(device), dataloader['train'].dataset[0][0].shape)
    else:
        print(model)    
    
    
def load_model(model, optimizer, exp_lr_scheduler, load_path='model.tar'):
    """
    Load model checkpoint from file.
    
    Args:
        model (nn.Module): Model to load weights into
        optimizer: Optimizer to load state into
        exp_lr_scheduler: Learning rate scheduler to load state into
        load_path (str): Path to checkpoint file. Default: 'model.tar'
        
    Returns:
        tuple: (model, optimizer, exp_lr_scheduler, epoch, val_loss)
    """
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    exp_lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint['current_score']
        
    return model, optimizer, exp_lr_scheduler, epoch, val_loss