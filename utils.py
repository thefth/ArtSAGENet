# -*- coding: utf-8 -*-
"""
Utility functions and classes for ArtSAGENet training.

Includes early stopping, multi-task blocks, model loading, and seed setting.
"""
import numpy as np
import os
import random
import torch
import torch.nn as nn


class EarlyStopping:
    """
    Early stopping to stop training when validation performance stops improving.
    
    Monitors validation loss or accuracy and stops training if no improvement
    is seen for a specified number of epochs (patience).
    """
    def __init__(self, accuracy=False, patience=10,
                 verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            accuracy (bool): If True, monitor accuracy (higher is better).
                If False, monitor loss (lower is better). Default: False
            patience (int): Number of epochs with no improvement before stopping.
                Default: 10
            verbose (bool): If True, prints messages when saving checkpoints.
                Default: False
            delta (float): Minimum change to qualify as an improvement.
                Default: 0
            path (str): Path for saving the best model checkpoint.
                Default: 'checkpoint.pt'
        """
        self.accuracy = accuracy
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_best = np.Inf if not accuracy else 0.0

    def __call__(self, epoch, current_score, model, optimizer, lr_scheduler):
        """
        Check if training should stop based on current validation score.
        
        Args:
            epoch (int): Current epoch number
            current_score (float): Current validation score (loss or accuracy)
            model (nn.Module): Model to save if score improves
            optimizer: Optimizer state to save
            lr_scheduler: Learning rate scheduler state to save
        """
        score = current_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, current_score, model, optimizer, lr_scheduler)
        elif (not self.accuracy and score >= self.best_score + self.delta) or \
             (self.accuracy and score <= self.best_score - self.delta):
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
        Save model checkpoint when validation score improves.
        
        Args:
            epoch (int): Current epoch number
            current_score (float): Current validation score
            model (nn.Module): Model to save
            optimizer: Optimizer state to save
            lr_scheduler: Learning rate scheduler state to save
        """
        if self.verbose:
            print(f'Validation score improved ({self.val_score_best:.6f} --> '
                  f'{current_score:.6f}). Saving model ...')
        
        self.val_score_best = current_score
        
        torch.save({
            'epoch': epoch,
            'current_score': current_score,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        }, self.path)


class Multitask_Block(nn.Module):
    """
    Multi-task learning block with separate output heads.
    
    Creates independent linear layers for multiple prediction tasks
    from shared features.
    """
    def __init__(self, num_in_features, num_classes_task1,
                 num_classes_task2, num_classes_task3):
        """
        Args:
            num_in_features (int): Input feature dimension
            num_classes_task1 (int): Number of classes for task 1
            num_classes_task2 (int): Number of classes for task 2
            num_classes_task3 (int): Number of classes for task 3
        """
        super(Multitask_Block, self).__init__()

        self.task1 = nn.Linear(num_in_features, num_classes_task1)
        self.task2 = nn.Linear(num_in_features, num_classes_task2)
        self.task3 = nn.Linear(num_in_features, num_classes_task3)

    def forward(self, x):
        """
        Forward pass through all task heads.
        
        Args:
            x (torch.Tensor): Shared features [batch_size, num_in_features]
            
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


def load_model(model, optimizer, exp_lr_scheduler, load_path='model.tar'):
    """
    Load a saved model checkpoint.
    
    Args:
        model (nn.Module): Model to load state into
        optimizer: Optimizer to load state into
        exp_lr_scheduler: Learning rate scheduler to load state into
        load_path (str): Path to the saved checkpoint
        
    Returns:
        tuple: (model, optimizer, exp_lr_scheduler, epoch, val_score)
            - model: Model with loaded weights
            - optimizer: Optimizer with loaded state
            - exp_lr_scheduler: Scheduler with loaded state
            - epoch: Epoch number from checkpoint
            - val_score: Validation score from checkpoint
    """
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    exp_lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    epoch = checkpoint['epoch']
    val_score = checkpoint['current_score']
   
    return model, optimizer, exp_lr_scheduler, epoch, val_score