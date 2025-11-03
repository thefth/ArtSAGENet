# -*- coding: utf-8 -*-
"""
Custom data loader for ArtSAGENet.

This module provides a customized neighbor sampler that extends PyTorch Geometric's
approach to additionally load and transform images. The implementation is adapted
from PyG v1.6.1's NeighborSampler but modified to handle image loading on-the-fly.

Note: This is a standalone implementation to ensure stability and compatibility.
"""
from typing import List, Optional, Tuple, NamedTuple
from PIL import Image

import torch
from torch_sparse import SparseTensor


class Adj(NamedTuple):
    """
    Named tuple for storing adjacency information in bipartite graphs.
    
    Attributes:
        edge_index (torch.Tensor): Edge indices of the bipartite graph
        e_id (torch.Tensor): IDs of original edges in the full graph
        size (Tuple[int, int]): Shape of the bipartite graph
    """
    edge_index: torch.Tensor
    e_id: torch.Tensor
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        """Move adjacency information to specified device."""
        return Adj(self.edge_index.to(*args, **kwargs),
                   self.e_id.to(*args, **kwargs), self.size)


class NeighborSamplerImages(torch.utils.data.DataLoader):
    """
    Neighbor sampler with image loading for hybrid CNN-GNN training.
    
    Adapted from PyTorch Geometric's NeighborSampler to additionally load
    and transform images for each sampled node. This enables joint training
    of CNN and GNN components.
    
    Based on "Inductive Representation Learning on Large Graphs"
    (Hamilton et al., NeurIPS 2017) <https://arxiv.org/abs/1706.02216>
    
    The sampler performs mini-batch training by:
    1. Sampling k-hop neighborhoods for a batch of target nodes
    2. Loading corresponding images from disk
    3. Applying image transformations
    4. Returning bipartite graphs and images for forward pass
    
    Args:
        list_ (List[str]): List of image file paths for each node in the graph
        image_transform (torchvision.transforms.Compose): Composition of
            image transformations to apply
        edge_index (LongTensor): Edge indices of the full graph [2, num_edges]
        sizes (List[int]): Number of neighbors to sample at each hop.
            If sizes[i] = -1, all neighbors are included at layer i
        node_idx (LongTensor, optional): Nodes to sample for creating mini-batches.
            If None, all nodes are considered
        num_nodes (int, optional): Total number of nodes in the graph.
            If None, inferred from edge_index
        flow (str, optional): Message passing direction.
            Either 'source_to_target' or 'target_to_source'.
            Default: 'source_to_target'
        **kwargs: Additional arguments for torch.utils.data.DataLoader
            (batch_size, shuffle, num_workers, etc.)
    """
    def __init__(self, list_: List[str], image_transform,
                 edge_index: torch.Tensor, sizes: List[int],
                 node_idx: Optional[torch.Tensor] = None,
                 num_nodes: Optional[int] = None,
                 flow: str = "source_to_target", **kwargs):
        
        self.list_ = list_
        self.image_transform = image_transform
        
        # Build sparse adjacency matrix
        N = int(edge_index.max() + 1) if num_nodes is None else num_nodes
        edge_attr = torch.arange(edge_index.size(1))
        adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                           value=edge_attr, sparse_sizes=(N, N),
                           is_sorted=False)
        adj = adj.t() if flow == 'source_to_target' else adj
        self.adj = adj.to('cpu')

        # Process node indices
        if node_idx is None:
            node_idx = torch.arange(N)
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        self.sizes = sizes
        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        super(NeighborSamplerImages, self).__init__(node_idx.tolist(),
                                                     collate_fn=self.sample,
                                                     **kwargs)

    def sample(self, batch):
        """
        Sample neighbors and load images for a batch of nodes.
        
        Args:
            batch: Batch of node indices
            
        Returns:
            tuple: (batch_size, n_id, imgs, adjs) where:
                - batch_size (int): Number of target nodes
                - n_id (torch.Tensor): All node IDs involved in computation
                - imgs (list): List of transformed images
                - adjs (list or Adj): Adjacency information for each layer
        """
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)
        adjs: List[Adj] = []

        # Sample neighbors layer by layer
        n_id = batch
        for size in self.sizes:
            adj, n_id = self.adj.sample_adj(n_id, size, replace=False)
            if self.flow == 'source_to_target':
                adj = adj.t()
            row, col, e_id = adj.coo()
            size = adj.sparse_sizes()
            edge_index = torch.stack([row, col], dim=0)

            adjs.append(Adj(edge_index, e_id, size))
        
        # Load and transform images for target nodes
        imgs = []
        for index in n_id[:batch_size]:
            img = Image.open(self.list_[index])
            
            if self.image_transform is not None:
                img = self.image_transform(img)
                
            imgs.append(img)
        
        # Return in reverse order for message passing
        if len(adjs) > 1:
            return batch_size, n_id, imgs, adjs[::-1]
        else:
            return batch_size, n_id, imgs, adjs[0]

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)