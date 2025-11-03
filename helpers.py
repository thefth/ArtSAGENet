# -*- coding: utf-8 -*-
"""
Helper functions for training and evaluating ArtSAGENet.

Provides training loops for both single-task and multi-task learning scenarios,
along with evaluation functions.
"""
import numpy as np
import torch
import copy
import utils

from sklearn.metrics import average_precision_score, classification_report
from datetime import datetime


def train_model(model, dataloaders_dict, features, labels, dataset_sizes,
                criterion, optimizer, scheduler, monitor_accuracy=False,
                task_type='classification', save_path=None,
                regression_threshold=0.03355705,
                device=torch.device('cpu'), num_epochs=50):
    """
    Train a single-task ArtSAGENet model.
    
    Args:
        model (nn.Module): ArtSAGENet model to train
        dataloaders_dict (dict): Dictionary with 'train', 'val', 'test' dataloaders
        features (torch.Tensor): Node features for all nodes in the graph
        labels (torch.Tensor): Labels for all nodes
        dataset_sizes (dict): Dictionary with dataset sizes for each split
        criterion: Loss function
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        monitor_accuracy (bool): If True, monitor accuracy; if False, monitor loss
        task_type (str): Type of task ('classification', 'regression', or 'retrieval')
        save_path (str, optional): Path to save best model. If None, model won't be saved
        regression_threshold (float): Threshold for regression accuracy calculation
        device (torch.device): Device to run training on
        num_epochs (int): Number of training epochs
        
    Returns:
        tuple: (model, final_val_loss, final_epoch)
    """
    since = datetime.now()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_metric = 0.0
    
    # Initialize early stopping only if save_path is provided
    if save_path:
        early_stopping = utils.EarlyStopping(path=save_path, 
                                             accuracy=monitor_accuracy, 
                                             patience=50, verbose=True)

    # Determine metric name for printing
    metric_name = 'Acc' if task_type != 'retrieval' else 'mAP'

    print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Starting training...')
    
    for epoch in range(num_epochs):
        print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
              f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 60)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            if task_type == 'retrieval':
                outputs = []
                targets = []
            
            # Iterate over batches
            for batch_size, n_id, imgs, adjs in dataloaders_dict[phase]:
                adjs = [adj.to(device) for adj in adjs]
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    out = model(torch.stack(imgs).to(device),
                                features[n_id], adjs)
            
                    if task_type != 'regression':
                        loss = criterion(out, labels[n_id[:batch_size]])
                    else:
                        loss = criterion(out.flatten(),
                                         labels[n_id[:batch_size]].float())
                    
                    # Backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Accumulate metrics
                running_loss += loss.item() * batch_size
                
                if task_type == 'classification':
                    running_corrects += torch.sum(torch.max(out, 1)[1] 
                                                  == labels[n_id[:batch_size]])
                elif task_type == 'regression':
                    running_corrects += torch.sum(torch.abs(out.flatten() 
                            - labels[n_id[:batch_size]].float()) <= regression_threshold)
                elif task_type == 'retrieval':
                    outputs.extend(out.sigmoid().cpu())
                    targets.extend(labels[n_id[:batch_size]].cpu())
                    
            epoch_loss = running_loss / dataset_sizes[phase]
            
            if task_type != 'retrieval':
                epoch_metric = running_corrects.double() / dataset_sizes[phase]
            else:
                outputs_ = torch.stack(outputs).detach().numpy()
                targets_ = torch.stack(targets).detach().numpy()
                epoch_metric = average_precision_score(targets_, outputs_,
                                                       average='macro') 
            
            # Handle validation phase
            if phase == 'val':
                if save_path:
                    if not monitor_accuracy:
                        scheduler.step(epoch_loss)
                        early_stopping(epoch=epoch, current_score=epoch_loss,
                                       model=model, optimizer=optimizer, 
                                       lr_scheduler=scheduler)
                    else:
                        scheduler.step(epoch_metric)
                        early_stopping(epoch=epoch, current_score=epoch_metric,
                                       model=model, optimizer=optimizer, 
                                       lr_scheduler=scheduler)
            
                    if early_stopping.early_stop:
                        print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
                              f'Early stopping')
                        return model, epoch_loss, epoch
                else:
                    scheduler.step(epoch_loss if not monitor_accuracy else epoch_metric)

            print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
                  f'{phase.capitalize()} Loss: {epoch_loss:.4f}, '
                  f'{metric_name}: {epoch_metric:.4f}')
                
            # Save best model
            if phase == 'val' and epoch_metric > best_metric:
                best_metric = epoch_metric
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = datetime.now() - since
    print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
          f'Training complete in {time_elapsed.seconds // 60}m '
          f'{time_elapsed.seconds % 60}s')
    print(f'Best val {metric_name}: {best_metric:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_loss, epoch


def train_model_multitask(model, dataloaders_dict, features, labels,
                           dataset_sizes, criterion, optimizer, scheduler,
                           monitor_accuracy=False, task_type='classification',
                           save_path=None, regression_threshold=0.03355705,
                           task_names=None,
                           device=torch.device('cpu'), num_epochs=50):
    """
    Train a multi-task ArtSAGENet model.
    
    Args:
        model (nn.Module): ArtSAGENet model to train (with multitask=True)
        dataloaders_dict (dict): Dictionary with 'train', 'val', 'test' dataloaders
        features (torch.Tensor): Node features for all nodes in the graph
        labels (list): List of [task1_labels, task2_labels, task3_labels]
        dataset_sizes (dict): Dictionary with dataset sizes for each split
        criterion (list): List of loss functions for each task
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        monitor_accuracy (bool): If True, monitor accuracy; if False, monitor loss
        task_type (str): Type of third task ('classification' or 'regression')
        save_path (str, optional): Path to save best model. If None, model won't be saved
        regression_threshold (float): Threshold for regression accuracy calculation
        task_names (list, optional): Names for the three tasks for printing.
            Default: ['Task1', 'Task2', 'Task3']
        device (torch.device): Device to run training on
        num_epochs (int): Number of training epochs
        
    Returns:
        tuple: (model, final_val_loss, final_epoch)
    """
    if task_names is None:
        task_names = ['Task1', 'Task2', 'Task3']
    
    since = datetime.now()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_metric = 0.0
    
    # Initialize early stopping only if save_path is provided
    if save_path:
        early_stopping = utils.EarlyStopping(path=save_path,
                                             accuracy=monitor_accuracy,
                                             patience=50, verbose=True)

    print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Starting training...')

    for epoch in range(num_epochs):
        print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
              f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 60)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            task1_corrects = 0
            task2_corrects = 0
            task3_corrects = 0
            
            # Iterate over batches
            for batch_size, n_id, imgs, adjs in dataloaders_dict[phase]:
                adjs = [adj.to(device) for adj in adjs]
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    out = model(torch.stack(imgs).to(device),
                                features[n_id], adjs)
                    
                    loss1 = criterion[0](out[0], labels[0][n_id[:batch_size]])
                    loss2 = criterion[1](out[1], labels[1][n_id[:batch_size]])
                    
                    if task_type == 'classification':
                        loss3 = criterion[2](out[2], labels[-1][n_id[:batch_size]])
                    else:
                        loss3 = criterion[2](out[2].flatten(),
                                             labels[-1][n_id[:batch_size]].float())
                    
                    loss = loss1 + loss2 + loss3
                    
                    # Backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Accumulate metrics
                running_loss += loss.item() * batch_size
                task1_corrects += torch.sum(torch.max(out[0], 1)[1] 
                                            == labels[0][n_id[:batch_size]])
                task2_corrects += torch.sum(torch.max(out[1], 1)[1] 
                                            == labels[1][n_id[:batch_size]])
                
                if task_type == 'classification':
                    task3_corrects += torch.sum(torch.max(out[-1], 1)[1]
                                                == labels[-1][n_id[:batch_size]])
                else:
                    task3_corrects += torch.sum(torch.abs(out[2].flatten()
                         - labels[-1][n_id[:batch_size]].float()) <= regression_threshold)
                    
            epoch_loss = running_loss / dataset_sizes[phase]
            task1_acc = task1_corrects.double() / dataset_sizes[phase]
            task2_acc = task2_corrects.double() / dataset_sizes[phase]
            task3_acc = task3_corrects.double() / dataset_sizes[phase]
            epoch_acc = (task1_acc + task2_acc + task3_acc) / 3
            
            # Handle validation phase
            if phase == 'val':
                if save_path:
                    if not monitor_accuracy:
                        scheduler.step(epoch_loss)
                        early_stopping(epoch=epoch, current_score=epoch_loss,
                                       model=model, optimizer=optimizer, 
                                       lr_scheduler=scheduler)
                    else:
                        scheduler.step(epoch_acc)
                        early_stopping(epoch=epoch, current_score=epoch_acc.item(),
                                       model=model, optimizer=optimizer, 
                                       lr_scheduler=scheduler)
            
                    if early_stopping.early_stop:
                        print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
                              f'Early stopping')
                        return model, epoch_loss, epoch
                else:
                    scheduler.step(epoch_loss if not monitor_accuracy else epoch_acc)
            
            print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
                  f'{phase.capitalize()} Loss: {epoch_loss:.4f}, '
                  f'{task_names[0]} Acc: {task1_acc:.4f}, '
                  f'{task_names[1]} Acc: {task2_acc:.4f}, '
                  f'{task_names[2]} Acc: {task3_acc:.4f}')

            # Save best model
            if phase == 'val' and task1_acc > best_metric:
                best_metric = task1_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = datetime.now() - since
    print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
          f'Training complete in {time_elapsed.seconds // 60}m '
          f'{time_elapsed.seconds % 60}s')
    print(f'Best val {task_names[0]} Acc: {best_metric:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_loss, epoch


def test_model(model, dataloaders_dict, features, labels, dataset_sizes,
               criterion, task_type='classification',
               regression_threshold=0.03355705,
               device=torch.device('cpu')):
    """
    Evaluate a single-task ArtSAGENet model on the test set.
    
    Args:
        model (nn.Module): Trained ArtSAGENet model
        dataloaders_dict (dict): Dictionary with 'test' dataloader
        features (torch.Tensor): Node features for all nodes
        labels (torch.Tensor): Labels for all nodes
        dataset_sizes (dict): Dictionary with dataset sizes
        criterion: Loss function
        task_type (str): Type of task ('classification', 'regression', or 'retrieval')
        regression_threshold (float): Threshold for regression accuracy
        device (torch.device): Device to run evaluation on
        
    Returns:
        tuple: (model, test_loss, test_metric)
    """
    since = datetime.now()
    print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Evaluating on test set...')
    
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    if task_type == 'retrieval':
        outputs = []
        targets = []

    # Iterate over test batches
    for batch_size, n_id, imgs, adjs in dataloaders_dict['test']:
        adjs = [adj.to(device) for adj in adjs]
        
        # Forward pass
        with torch.no_grad():
            out = model(torch.stack(imgs).to(device), features[n_id], adjs)
            
            if task_type != 'regression':
                loss = criterion(out, labels[n_id[:batch_size]])
            else:
                loss = criterion(out.flatten(),
                                 labels[n_id[:batch_size]].float())
            
            running_loss += loss.item() * batch_size
            
            if task_type == 'classification':
                running_corrects += torch.sum(torch.max(out, 1)[1] 
                                              == labels[n_id[:batch_size]])
            elif task_type == 'regression':
                running_corrects += torch.sum(torch.abs(out.flatten() 
                        - labels[n_id[:batch_size]].float()) <= regression_threshold)
            elif task_type == 'retrieval':
                outputs.extend(out.sigmoid().cpu())
                targets.extend(labels[n_id[:batch_size]].cpu())
            
    test_loss = running_loss / dataset_sizes['test']
    
    if task_type != 'retrieval':
        test_metric = running_corrects.double() / dataset_sizes['test']
        metric_name = 'Acc'
    else:
        outputs_ = torch.stack(outputs).detach().numpy()
        targets_ = torch.stack(targets).detach().numpy()
        
        outputs_threshold = outputs_.copy()
        outputs_threshold[outputs_ >= 0.5] = 1
        outputs_threshold[outputs_ < 0.5] = 0 
        
        clf = classification_report(targets_, outputs_threshold,
                                    zero_division=0, output_dict=True)
        
        print(['O' + k[0].upper() + ': ' + str(np.round(v, 4))
               for k, v in clf['micro avg'].items() if k != 'support'])
        print(['C' + k[0].upper() + ': ' + str(np.round(v, 4))
               for k, v in clf['macro avg'].items() if k != 'support'])

        test_metric = average_precision_score(targets_, outputs_, average='macro')
        metric_name = 'mAP'
            
    print()
    time_elapsed = datetime.now() - since
    print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
          f'Testing complete in {time_elapsed.seconds // 60}m '
          f'{time_elapsed.seconds % 60}s')
    print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
          f'Test {metric_name}: {test_metric:.4f}')
    
    return model, test_loss, test_metric


def test_model_multitask(model, dataloaders_dict, features, labels,
                          dataset_sizes, criterion, task_type='classification',
                          regression_threshold=0.03355705,
                          task_names=None,
                          device=torch.device('cpu')):
    """
    Evaluate a multi-task ArtSAGENet model on the test set.
    
    Args:
        model (nn.Module): Trained ArtSAGENet model (with multitask=True)
        dataloaders_dict (dict): Dictionary with 'test' dataloader
        features (torch.Tensor): Node features for all nodes
        labels (list): List of [task1_labels, task2_labels, task3_labels]
        dataset_sizes (dict): Dictionary with dataset sizes
        criterion (list): List of loss functions for each task
        task_type (str): Type of third task ('classification' or 'regression')
        regression_threshold (float): Threshold for regression accuracy
        task_names (list, optional): Names for the three tasks.
            Default: ['Task1', 'Task2', 'Task3']
        device (torch.device): Device to run evaluation on
        
    Returns:
        tuple: (model, test_loss, task1_acc, task2_acc, task3_acc)
    """
    if task_names is None:
        task_names = ['Task1', 'Task2', 'Task3']
    
    since = datetime.now()
    print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Evaluating on test set...')
    
    model.eval()
    running_loss = 0.0
    task1_corrects = 0
    task2_corrects = 0
    task3_corrects = 0
    
    # Iterate over test batches
    for batch_size, n_id, imgs, adjs in dataloaders_dict['test']:
        adjs = [adj.to(device) for adj in adjs]
        
        # Forward pass
        with torch.no_grad():
            out = model(torch.stack(imgs).to(device), features[n_id], adjs)
            
            loss1 = criterion[0](out[0], labels[0][n_id[:batch_size]])
            loss2 = criterion[1](out[1], labels[1][n_id[:batch_size]])
            
            if task_type == 'classification':
                loss3 = criterion[2](out[2], labels[-1][n_id[:batch_size]])
            else:
                loss3 = criterion[2](out[2].flatten(),
                                     labels[-1][n_id[:batch_size]].float())
            
            loss = loss1 + loss2 + loss3
            
        running_loss += loss.item() * batch_size
        task1_corrects += torch.sum(torch.max(out[0], 1)[1] 
                                    == labels[0][n_id[:batch_size]])
        task2_corrects += torch.sum(torch.max(out[1], 1)[1] 
                                    == labels[1][n_id[:batch_size]])
        
        if task_type == 'classification':
            task3_corrects += torch.sum(torch.max(out[-1], 1)[1]
                                        == labels[-1][n_id[:batch_size]])
        else:
            task3_corrects += torch.sum(torch.abs(out[2].flatten()
                 - labels[-1][n_id[:batch_size]].float()) <= regression_threshold)
                
    test_loss = running_loss / dataset_sizes['test']
    task1_acc = task1_corrects.double() / dataset_sizes['test']
    task2_acc = task2_corrects.double() / dataset_sizes['test']
    task3_acc = task3_corrects.double() / dataset_sizes['test']
    
    print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
          f'Test Loss: {test_loss:.4f}, '
          f'{task_names[0]} Acc: {task1_acc:.4f}, '
          f'{task_names[1]} Acc: {task2_acc:.4f}, '
          f'{task_names[2]} Acc: {task3_acc:.4f}')

    time_elapsed = datetime.now() - since
    print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
          f'Testing complete in {time_elapsed.seconds // 60}m '
          f'{time_elapsed.seconds % 60}s')
    
    return model, test_loss, task1_acc, task2_acc, task3_acc