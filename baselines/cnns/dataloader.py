# -*- coding: utf-8 -*-
"""
Dataset loader for CNN baselines.

This module provides a PyTorch Dataset class for loading painting images
and their associated labels for single-task or multi-task learning.
"""
import torch
from PIL import Image


class load_dataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset for loading painting images with labels.
    
    Supports both single-task and multi-task learning configurations.
    """
    def __init__(self, list_, image_dir='WikiArt/Dataset/',
                 transform=None, multitask=False):
        """
        Args:
            list_ (list): List containing image paths and labels.
                         For single-task: [image_paths, labels]
                         For multi-task: [image_paths, task1_labels, task2_labels, task3_labels]
            image_dir (string): Path to the images directory.
            transform (callable, optional): Optional transform to be applied on a sample.
            multitask (bool): If True, returns labels for multiple tasks.
            
        Returns:
            tuple: (image, labels) where labels is a list of torch tensors
        """
        self.list_ = list_
        self.image_dir = image_dir
        self.transform = transform
        self.multitask = multitask
        
    def __getitem__(self, index):
        """
        Get a single sample from the dataset.
        
        Args:
            index (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image, list_of_labels)
        """
        img = Image.open(self.image_dir + self.list_[0][index])
        
        if self.transform is not None:
            img = self.transform(img)

        if not self.multitask:
            y = self.list_[1][index] 
            list_of_labels = [torch.from_numpy(y)]
        else:
            task_1 = self.list_[1][index] 
            task_2 = self.list_[2][index] 
            task_3 = self.list_[3][index] 
        
            list_of_labels = [torch.from_numpy(task_1),
                              torch.from_numpy(task_2),
                              torch.from_numpy(task_3)]
            
        return img, list_of_labels
            
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.list_[0])