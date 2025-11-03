# -*- coding: utf-8 -*-
"""
ArtSAGENet: Graph Neural Networks for Knowledge Enhanced Visual Representation of Paintings.

This module implements the ArtSAGENet architecture that combines:
- ResNet-152 CNN for visual feature extraction
- GraphSAGE layers for graph-based knowledge propagation
- Multiple merge strategies for fusing visual and graph features
- Support for single-task and multi-task learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torch_geometric.nn import SAGEConv


class ArtSAGENet(nn.Module):
    """
    ArtSAGENet model combining CNN visual features with GNN graph features.
    
    The model consists of:
    1. ResNet-152 backbone for extracting visual features from images
    2. Two-layer GraphSAGE for propagating knowledge through the graph
    3. Fusion layer combining visual and graph representations
    4. Task-specific output heads for classification/regression
    
    Attributes:
        dropout (float): Dropout rate for regularization
        merge (str): Strategy for merging visual and graph features
        multitask (bool): Whether to use multi-task learning
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 dropout=0.5, fine_tune=True, 
                 merge='concatenate', multitask=True):
        """
        Initialize ArtSAGENet model.
        
        Args:
            in_channels (int): Number of input node features
            hidden_channels (int): Number of hidden units in GNN layers
            out_channels (int or list): Number of output classes
                                       - int for single-task
                                       - list of 3 ints for multi-task
            dropout (float): Dropout probability. Default: 0.5
            fine_tune (bool): If True, load pretrained ImageNet weights. Default: True
            merge (str): Feature fusion strategy. Options:
                        - 'concatenate': Concatenate visual and graph features
                        - 'add': Element-wise addition
                        - 'multiply': Element-wise multiplication  
                        - 'mean': Average of features
                        Default: 'concatenate'
            multitask (bool): If True, use multi-task learning. Default: True
        """
        super(ArtSAGENet, self).__init__()
        
        self.dropout = dropout
        self.merge = merge
        self.multitask = multitask
        
        # CNN backbone: ResNet-152
        self.cnn = models.resnet152(pretrained=fine_tune)
        self.features = nn.Sequential(
            self.cnn.conv1,
            self.cnn.bn1,
            self.cnn.relu,
            self.cnn.maxpool,
            self.cnn.layer1,
            self.cnn.layer2,
            self.cnn.layer3,
            self.cnn.layer4,
        )
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        # GraphSAGE layers
        self.num_layers = 2
        self.gnn = nn.ModuleList()
        self.gnn.append(SAGEConv(in_channels, hidden_channels))
        self.gnn.append(SAGEConv(hidden_channels, hidden_channels))
        
        # Task-specific output heads
        if self.multitask:
            # Multi-task learning: separate head for each task
            if self.merge == 'concatenate':
                self.task1 = nn.Linear(hidden_channels * 2, out_channels[0])
                self.task2 = nn.Linear(hidden_channels * 2, out_channels[1])
                self.task3 = nn.Linear(hidden_channels * 2, out_channels[-1])
            else:
                self.task1 = nn.Linear(hidden_channels, out_channels[0])
                self.task2 = nn.Linear(hidden_channels, out_channels[1])
                self.task3 = nn.Linear(hidden_channels, out_channels[-1])
        else:
            # Single-task learning
            if self.merge == 'concatenate':
                self.task1 = nn.Linear(hidden_channels * 2, out_channels)
            else:
                self.task1 = nn.Linear(hidden_channels, out_channels)

    def forward(self, image, node_features, adjs):
        """
        Forward pass through ArtSAGENet.
        
        Args:
            image (torch.Tensor): Batch of images [batch_size, 3, H, W]
            node_features (torch.Tensor): Node features from sampled neighborhood
            adjs (list): List of (edge_index, e_id, size) tuples from neighbor sampler
            
        Returns:
            torch.Tensor or tuple: Model predictions
                - Single-task: tensor of shape [batch_size, num_classes]
                - Multi-task: tuple of 3 tensors for each task
        """
        # Extract visual features using CNN
        visual_features = self.features(image)
        visual_features = self.pooling(visual_features)
        visual_features = visual_features.view(visual_features.size(0), -1)
        
        # Propagate through GraphSAGE layers
        for i, (edge_index, _, size) in enumerate(adjs):
            target_nodes = node_features[:size[1]]  # Target nodes are always placed first
            node_features = self.gnn[i]((node_features, target_nodes), edge_index)
            if i != self.num_layers - 1:
                node_features = F.relu(node_features)
                node_features = F.dropout(node_features, p=self.dropout, training=self.training)
        
        # Merge visual and graph features
        if self.merge == 'concatenate':
            merged = torch.cat((visual_features, node_features), dim=1)
        elif self.merge == 'add':
            merged = visual_features + node_features
        elif self.merge == 'multiply':
            merged = visual_features * node_features
        elif self.merge == 'mean':
            merged = torch.stack((visual_features, node_features))
            merged = torch.mean(merged, dim=0)

        # Task-specific predictions
        if self.multitask:
            task1 = self.task1(merged)
            task2 = self.task2(merged)
            task3 = self.task3(merged)
            return task1, task2, task3
        else:
            task1 = self.task1(merged)
            return task1