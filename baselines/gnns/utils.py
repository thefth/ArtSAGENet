# -*- coding: utf-8 -*-
"""
Utility functions for GNN baselines.

This module provides utility classes and functions including:
- Early stopping mechanism for GNN training
- Seed setting for reproducibility
"""
import numpy as np
import os
import random
import torch


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.
    
    Monitors validation loss and stops training if it doesn't improve
    after a given patience period.
    """
    def __init__(self, patience=10, verbose=False, delta=0,
                 path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): Number of epochs with no improvement before stopping.
                          Default: 10
            verbose (bool): If True, prints messages when saving checkpoints.
                          Default: False
            delta (float): Minimum change to qualify as an improvement.
                         Default: 0
            path (str): Path for saving the checkpoint.
                      Default: 'checkpoint.pt'
            trace_func (function): Function for printing messages.
                                 Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, epoch, val_loss, model, optimizer, lr_scheduler=None):
        """
        Check if training should stop based on validation loss.
        
        Args:
            epoch (int): Current epoch number
            val_loss (float): Current validation loss
            model (nn.Module): Model to save if loss improves
            optimizer: Optimizer state to save
            lr_scheduler: Learning rate scheduler state to save (optional)
        """
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, model, optimizer, lr_scheduler)
        elif score > self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, model, optimizer, lr_scheduler)
            self.counter = 0

    def save_checkpoint(self, epoch, val_loss, model, optimizer, lr_scheduler=None):
        """
        Save model checkpoint when validation loss improves.
        
        Args:
            epoch (int): Current epoch number
            val_loss (float): Current validation loss
            model (nn.Module): Model to save
            optimizer: Optimizer state to save
            lr_scheduler: Learning rate scheduler state to save (optional)
        """
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        
        self.val_loss_min = val_loss
        
        checkpoint = {
            'epoch': epoch,
            'val_loss': val_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        
        if lr_scheduler:
            checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
        
        torch.save(checkpoint, self.path)


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