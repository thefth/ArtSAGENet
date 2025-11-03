# -*- coding: utf-8 -*-
"""
Helper functions for CNN baselines.

This module provides core functionality for:
- Dataset creation with data augmentation
- Model initialization (AlexNet, VGG, ResNet)
- Training and testing procedures for single-task and multi-task learning
"""
import numpy as np
import sys
import torch
import torch.nn as nn
import copy
import utils

from dataloader import load_dataset
from torchvision import models, transforms
from sklearn.metrics import average_precision_score, classification_report
from datetime import datetime


def create_dataset(data, image_dir, task=1,
                   batch_size=16, train_shuffle=True,
                   val_test_shuffle=False, num_workers=0,
                   multitask=True):
    """
    Create PyTorch dataloaders with data augmentation for train/val/test splits.
    
    Args:
        data (dict): Dictionary containing 'train', 'val', and 'test' splits
        image_dir (str): Directory containing the images
        task (int): Task index for single-task learning (1, 2, or 3)
        batch_size (int): Batch size for dataloaders
        train_shuffle (bool): Whether to shuffle training data
        val_test_shuffle (bool): Whether to shuffle validation/test data
        num_workers (int): Number of workers for data loading
        multitask (bool): If True, return labels for all tasks
        
    Returns:
        tuple: (dataloaders_dict, dataset_sizes)
    """
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
    
    print('[{}] Transformations declared ...'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    
    if not multitask:
        train_lists = [[i[0] for i in data['train']], [i[task] for i in data['train']]]
        val_lists = [[i[0] for i in data['val']], [i[task] for i in data['val']]]
        test_lists = [[i[0] for i in data['test']], [i[task] for i in data['test']]]
    else:
        X_train, y1_train, y2_train, y3_train = [i[0] for i in data['train']],\
                                                [i[1] for i in data['train']],\
                                                [i[2] for i in data['train']],\
                                                [i[3] for i in data['train']]
        train_lists = [X_train, y1_train, y2_train, y3_train]
        
        X_val, y1_val, y2_val, y3_val = [i[0] for i in data['val']],\
                                        [i[1] for i in data['val']],\
                                        [i[2] for i in data['val']],\
                                        [i[3] for i in data['val']]
        val_lists = [X_val, y1_val, y2_val, y3_val]
        
        X_test, y1_test, y2_test, y3_test = [i[0] for i in data['test']],\
                                            [i[1] for i in data['test']],\
                                            [i[2] for i in data['test']],\
                                            [i[3] for i in data['test']]
        test_lists = [X_test, y1_test, y2_test, y3_test]
    
    training_dataset = load_dataset(list_=train_lists,
                                    image_dir=image_dir,
                                    transform=data_transforms['train'],
                                    multitask=multitask)
    
    val_dataset = load_dataset(list_=val_lists,
                               image_dir=image_dir,
                               transform=data_transforms['val'],
                               multitask=multitask)
    
    test_dataset = load_dataset(list_=test_lists,
                                image_dir=image_dir,
                                transform=data_transforms['test'],
                                multitask=multitask)
    
    dataloaders_dict = {'train': torch.utils.data.DataLoader(training_dataset,
                                                        batch_size=batch_size, 
                                                        shuffle=train_shuffle,
                                                        num_workers=num_workers),
                       'val': torch.utils.data.DataLoader(val_dataset,
                                                        batch_size=batch_size, 
                                                        shuffle=val_test_shuffle,
                                                        num_workers=num_workers),
                       'test': torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=batch_size, 
                                                        shuffle=val_test_shuffle,
                                                        num_workers=num_workers)
                       }
    
    dataset_sizes = {'train': len(train_lists[0]),
                     'val': len(val_lists[0]),
                     'test': len(test_lists[0])}
    
    return dataloaders_dict, dataset_sizes


def get_model(model_path=None, 
              model='resnet-152', 
              fine_tune=True, 
              multitask=False, 
              task_type='classification',
              mask=-100, 
              num_classes_task1=20,
              num_classes_task2=750, 
              num_classes_task3=13,
              lr=0.0001, 
              momentum=0.9,
              monitor_accuracy=False, 
              scheduler_type='plateau',
              device=torch.device('cpu')):
    """
    Initialize a CNN model with specified architecture and configuration.
    
    Args:
        model_path (str): Path to pretrained model (currently unused)
        model (str): Model architecture ('alexnet', 'vgg-16bn', 'vgg-19bn', 'resnet-34', 'resnet-152')
        fine_tune (bool): If True, load pretrained ImageNet weights and freeze most layers
        multitask (bool): If True, use multi-task learning head
        task_type (str): Type of task ('classification', 'regression', 'retrieval')
        mask (int): Mask value for loss computation
        num_classes_task1 (int): Number of classes for task 1
        num_classes_task2 (int): Number of classes for task 2
        num_classes_task3 (int): Number of classes for task 3
        lr (float): Learning rate
        momentum (float): Momentum for SGD optimizer
        monitor_accuracy (bool): If True, monitor accuracy for scheduler/early stopping
        scheduler_type (str): Type of learning rate scheduler ('multistep' or 'plateau')
        device (torch.device): Device to use for training
        
    Returns:
        tuple: (model_conv, criterion, optimizer_conv, exp_lr_scheduler)
    """
    # ---------- Backbone ----------
    if model in ['alexnet', 'vgg-16bn', 'vgg-19bn']:
        if model == 'alexnet':
            model_conv = models.alexnet(pretrained=fine_tune)
        elif model == 'vgg-16bn':
            model_conv = models.vgg16_bn(pretrained=fine_tune)
        else:
            model_conv = models.vgg19_bn(pretrained=fine_tune)

        if multitask:
            model_conv.classifier[-1] = utils.Multitask_Block(model_conv.classifier[-1].in_features,
                                                        num_classes_task1,
                                                        num_classes_task2,
                                                        num_classes_task3)
        else:
            model_conv.classifier[-1] = nn.Linear(model_conv.classifier[-1].in_features,
                                             num_classes_task1)
            
    elif model in ['resnet-34', 'resnet-152']:
        if model == 'resnet-34':
            model_conv = models.resnet34(pretrained=fine_tune)
        else:
            model_conv = models.resnet152(pretrained=fine_tune)
            
        if multitask:
            model_conv.fc = utils.Multitask_Block(model_conv.fc.in_features,
                                        num_classes_task1,
                                        num_classes_task2,
                                        num_classes_task3)
        else:
            model_conv.fc = nn.Linear(model_conv.fc.in_features,
                                      num_classes_task1)
    else:
        sys.exit('Not a valid model')
        
    # ---------- Freeze / unfreeze ----------
    if fine_tune:
        for param in model_conv.parameters():
            param.requires_grad = False 
            
    if model == 'alexnet':
        for param in model_conv.features[10].parameters():
            param.requires_grad = True

        for param in model_conv.classifier.parameters():
            param.requires_grad = True

        params = list(model_conv.features[10].parameters()) +\
                 list(model_conv.classifier.parameters())
        
    elif 'vgg' in model:
        for param in model_conv.features[-4:].parameters():
            param.requires_grad = True
        
        for param in model_conv.classifier.parameters():
            param.requires_grad = True
        
        params = list(model_conv.features[-4:].parameters()) +\
                 list(model_conv.classifier.parameters())
    
    else:
        for param in model_conv.layer4.parameters():
            param.requires_grad = True
            
        for param in model_conv.fc.parameters():
            param.requires_grad = True
            
        params = list(model_conv.layer4.parameters()) +\
                 list(model_conv.fc.parameters())
                                 
    model_conv = model_conv.to(device)
    
    # ---------- Loss ----------
    if task_type == 'classification':
        if multitask:
            criterion = [nn.CrossEntropyLoss(),
                         nn.CrossEntropyLoss(ignore_index=mask),
                         nn.CrossEntropyLoss()]
        else:
            criterion = [nn.CrossEntropyLoss(ignore_index=mask)]
                
    elif task_type == 'regression':
        if multitask:
            criterion = [nn.CrossEntropyLoss(),
                         nn.CrossEntropyLoss(ignore_index=mask),
                         nn.L1Loss()]
        else:
            criterion = [nn.L1Loss()]
           
    elif task_type == 'retrieval':
        criterion = [nn.BCEWithLogitsLoss()]
    
    # ---------- Optimizer ----------
    if fine_tune:
        optimizer_conv = torch.optim.SGD(params, lr=lr, momentum=momentum)
    else:
        optimizer_conv = torch.optim.SGD(model_conv.parameters(), lr=lr,
                                         momentum=momentum, weight_decay=0.0001)
    
    # ---------- Scheduler ----------
    if scheduler_type == 'multistep':
        exp_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_conv, milestones=[15, 30, 40], gamma=0.1, verbose=True)
    elif scheduler_type == 'plateau':
        exp_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_conv,
                                        'min' if not monitor_accuracy else 'max',
                                        patience=5,
                                        verbose=True)
    else:
        exp_lr_scheduler = None
    
    return model_conv, criterion, optimizer_conv, exp_lr_scheduler

  
def train_model(model, dataloaders_dict, dataset_sizes,
                criterion, optimizer, scheduler, monitor_accuracy=False,
                task_type='classification', save_path='checkpoint.pt',
                regression_threshold=0.03355705,
                device=torch.device('cpu'), num_epochs=25):
    """
    Train a single-task model.
    
    Args:
        model (nn.Module): Model to train
        dataloaders_dict (dict): Dictionary of dataloaders
        dataset_sizes (dict): Dictionary of dataset sizes
        criterion (list): List containing loss function(s)
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        monitor_accuracy (bool): If True, use accuracy for scheduling/early stopping
        task_type (str): Type of task ('classification', 'regression', 'retrieval')
        save_path (str): Path to save best model
        device (torch.device): Device for training
        num_epochs (int): Number of training epochs
        
    Returns:
        tuple: (model, epoch_loss, epoch)
    """
    since = datetime.now()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    early_stopping = utils.EarlyStopping(save_path=save_path, accuracy=monitor_accuracy,
                                        patience=50, verbose=True)
    
    for epoch in range(num_epochs):
        print('[{}] Epoch {}/{}'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), 
                                        epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            if task_type == 'retrieval':
                outputs_ = []
                targets_ = []

            # Iterate over data
            for inputs, labels in dataloaders_dict[phase]:
                inputs = inputs.to(device)
                labels = labels[0].to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    if task_type == 'classification':
                        loss = criterion[0](outputs,
                                           torch.max(labels.float(), 1)[1])
                    elif task_type == 'regression':
                        loss = criterion[0](outputs.flatten(), labels.float())
                    elif task_type == 'retrieval':
                        loss = criterion[0](outputs, labels)
                        
                    # Backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Loss
                running_loss += loss.item() * inputs.size(0)
                
                # Accuracy
                if task_type == 'classification':
                    running_corrects += torch.sum(preds == torch.max(labels, 1)[1])
                elif task_type == 'regression':
                    running_corrects += torch.sum(torch.abs(outputs.flatten() - labels)
                                                  <= regression_threshold)
                elif task_type == 'retrieval':
                    outputs_.extend(outputs.sigmoid().cpu())
                    targets_.extend(labels.cpu())
                
            epoch_loss = running_loss / dataset_sizes[phase]
            
            if task_type != 'retrieval':
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
            else:
                outputs_ = torch.stack(outputs_)
                targets_ = torch.stack(targets_)
                epoch_acc = average_precision_score(targets_.detach().numpy(),
                                                    outputs_.detach().numpy(),
                                                    average='macro')
            
            if phase == 'val':
                if not monitor_accuracy:
                    scheduler.step(epoch_loss)
                    early_stopping(epoch=epoch, current_score=epoch_loss,
                                   model=model, optimizer=optimizer, 
                                   lr_scheduler=scheduler)
                else:
                    scheduler.step(epoch_acc)
                    early_stopping(epoch=epoch, current_score=epoch_acc,
                                   model=model, optimizer=optimizer, 
                                   lr_scheduler=scheduler)
        
                if early_stopping.early_stop:
                    print("[{}] Early stopping".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
                    return model, epoch_loss, epoch

            if task_type != 'retrieval':
                print('[{}] {} Loss: {:.4f} Acc: {:.4f}'.format(
                    datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                    phase, epoch_loss, epoch_acc))
            else:
                print('[{}] {} Loss: {:.4f} mAP: {:.4f}'.format(
                    datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                    phase, epoch_loss, epoch_acc))
                
            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = datetime.now() - since
    print('[{}] Training complete in {:.0f}m {:.0f}s'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        time_elapsed.seconds // 60, time_elapsed.seconds % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_loss, epoch


def train_model_multitask(model, dataloaders_dict, dataset_sizes,
                          criterion, optimizer, scheduler, monitor_accuracy=False,
                          task_type='classification', save_path='checkpoint.pt',
                          regression_threshold=0.03355705,
                          device=torch.device('cpu'), num_epochs=25):
    """
    Train a multi-task model.
    
    Args:
        model (nn.Module): Model to train
        dataloaders_dict (dict): Dictionary of dataloaders
        dataset_sizes (dict): Dictionary of dataset sizes
        criterion (list): List of loss functions for each task
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        monitor_accuracy (bool): If True, use accuracy for scheduling/early stopping
        task_type (str): Type of task for task 3 ('classification' or 'regression')
        save_path (str): Path to save best model
        device (torch.device): Device for training
        num_epochs (int): Number of training epochs
        
    Returns:
        tuple: (model, epoch_loss, epoch)
    """
    since = datetime.now()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    early_stopping = utils.EarlyStopping(save_path=save_path,
                                        accuracy=monitor_accuracy,
                                        patience=50, verbose=True)

    for epoch in range(num_epochs):
        print('[{}] Epoch {}/{}'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                                        epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            task1_corrects = 0.0
            task2_corrects = 0.0
            task3_corrects = 0.0
            
            for inputs, tasks in dataloaders_dict[phase]:
                inputs = inputs.to(device) 
                task1 = tasks[0].to(device)
                task2 = tasks[1].to(device)
                task3 = tasks[-1].to(device)

                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss1 = criterion[0](outputs[0],
                                         torch.max(task1.float(), 1)[1])
                    loss2 = criterion[1](outputs[1],
                                         torch.max(task2.float(), 1)[1])
                    
                    if task_type == 'classification':
                        loss3 = criterion[2](outputs[2],
                                             torch.max(task3.float(), 1)[1])
                    elif task_type == 'regression':
                        loss3 = criterion[2](outputs[2].flatten(),
                                             task3.float())
                    
                    if task_type == 'regression':
                        loss = loss1 + loss2 + (1 * loss3)
                    else:
                        loss = loss1 + loss2 + loss3

                    # Backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                task1_corrects += torch.sum(torch.max(outputs[0], 1)[1]
                                            == torch.max(task1, 1)[1])
                task2_corrects += torch.sum(torch.max(outputs[1], 1)[1] 
                                            == torch.max(task2, 1)[1])
                                              
                if task_type == 'classification':
                    task3_corrects += torch.sum(torch.max(outputs[2], 1)[1]
                                                == torch.max(task3, 1)[1])
                elif task_type == 'regression':
                    task3_corrects += torch.sum(torch.abs(outputs[2].flatten()
                                                - task3.float()) <= regression_threshold)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            task1_acc = task1_corrects.double() / dataset_sizes[phase]
            task2_acc = task2_corrects.double() / dataset_sizes[phase]
            task3_acc = task3_corrects.double() / dataset_sizes[phase]
            
            if phase == 'val':
                if not monitor_accuracy:
                    scheduler.step(epoch_loss)
                    early_stopping(epoch=epoch, current_score=epoch_loss,
                                   model=model, optimizer=optimizer, 
                                   lr_scheduler=scheduler)
                else:
                    epoch_acc = task1_acc + task2_acc + task3_acc
                    scheduler.step(epoch_acc.item())
                    early_stopping(epoch=epoch, current_score=epoch_acc.item(),
                                   model=model, optimizer=optimizer, 
                                   lr_scheduler=scheduler)
        
                if early_stopping.early_stop:
                    print("Early stopping")
                    return model, epoch_loss, epoch
            
            print('[{}] {} Loss: {:.4f} Style Acc: {:.4f} Artist Acc: {:.4f} Date Acc: {:.4f}'.format(
                datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                phase, epoch_loss, task1_acc, task2_acc, task3_acc))

            # Deep copy the model
            if phase == 'val' and task1_acc > best_acc:
                best_acc = task1_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = datetime.now() - since
    print('[{}] Training complete in {:.0f}m {:.0f}s'.format(
        datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        time_elapsed.seconds // 60, time_elapsed.seconds % 60))
    print('Best val Acc: {:4f}'.format(task1_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_loss, epoch


def test_model(model, dataloaders_dict, 
               dataset_sizes, criterion,
               optimizer, task_type='classification',
               regression_threshold=0.03355705, device=torch.device('cpu')):
    """
    Test a single-task model on the test set.
    
    Args:
        model (nn.Module): Trained model
        dataloaders_dict (dict): Dictionary of dataloaders
        dataset_sizes (dict): Dictionary of dataset sizes
        criterion: Loss function
        optimizer: Optimizer
        task_type (str): Type of task ('classification', 'regression', 'retrieval')
        regression_threshold (float): Threshold for regression accuracy
        device (torch.device): Device for testing
        
    Returns:
        tuple: (model, test_loss, test_accuracy)
    """
    since = datetime.now()
    print('Testing......')
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    if task_type == 'retrieval':
        outputs_ = []
        targets_ = []

    for inputs, labels in dataloaders_dict['test']:
        inputs = inputs.to(device)
        labels = labels[0].to(device)
        
        optimizer.zero_grad()

        # Forward pass
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            if task_type == 'classification':
                loss = criterion[0](outputs, torch.max(labels.float(), 1)[1])
            elif task_type == 'regression':
                loss = criterion[0](outputs.flatten(), labels.float())
            elif task_type == 'retrieval':
                loss = criterion[0](outputs, labels)
            
        # Loss
        running_loss += loss.item() * inputs.size(0)
        
        if task_type == 'classification':
            running_corrects += torch.sum(preds == torch.max(labels, 1)[1])
        elif task_type == 'regression':
            running_corrects += torch.sum(torch.abs(outputs.flatten() - labels)
                                          <= regression_threshold)
        elif task_type == 'retrieval':
            outputs_.extend(outputs.sigmoid().cpu())
            targets_.extend(labels.cpu())
        
    epoch_loss = running_loss / dataset_sizes['test']
    
    if task_type != 'retrieval':
        epoch_acc = running_corrects.double() / dataset_sizes['test']
        print('[{}]  Loss: {:.4f} Acc: {:.4f}'.format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            epoch_loss, epoch_acc))
    else:
        outputs_ = torch.stack(outputs_)
        targets_ = torch.stack(targets_)
        
        outputs_threshold = outputs_.detach().numpy().copy()
        outputs_threshold[outputs_ >= 0.5] = 1
        outputs_threshold[outputs_ < 0.5] = 0 

        clf = classification_report(targets_,
                                    outputs_threshold, output_dict=True)
        print(['O' + k[0].upper() + ': ' + str(np.round(v, 4)) 
               for k, v in clf['micro avg'].items() if k != 'support'])
        print(['C' + k[0].upper() + ': ' + str(np.round(v, 4))
               for k, v in clf['macro avg'].items() if k != 'support'])
            
        epoch_acc = average_precision_score(targets_.detach().numpy(),
                                            outputs_.detach().numpy(),
                                            average='macro')
        print('[{}]  Loss: {:.4f} mAP: {:.4f}'.format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            epoch_loss, epoch_acc))

    print()
    
    time_elapsed = datetime.now() - since
    print('[{}] Testing complete in {:.0f}m {:.0f}s'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
    time_elapsed.seconds // 60, time_elapsed.seconds % 60))
    
    return model, epoch_loss, epoch_acc


def test_model_multitask(model, dataloaders_dict, dataset_sizes,
                         criterion, optimizer, 
                         task_type='classification', regression_threshold=0.03355705,
                         device=torch.device('cuda')):
    """
    Test a multi-task model on the test set.
    
    Args:
        model (nn.Module): Trained model
        dataloaders_dict (dict): Dictionary of dataloaders
        dataset_sizes (dict): Dictionary of dataset sizes
        criterion (list): List of loss functions for each task
        optimizer: Optimizer
        task_type (str): Type of task for task 3 ('classification' or 'regression')
        regression_threshold (float): Threshold for regression accuracy (task 3)
        device (torch.device): Device for testing
        
    Returns:
        tuple: (model, test_loss, task1_acc, task2_acc, task3_acc)
    """
    since = datetime.now()
    print('Testing.......')
    model.eval()
    running_loss = 0.0
    task1_corrects = 0.0
    task2_corrects = 0.0
    task3_corrects = 0.0
    
    for inputs, tasks in dataloaders_dict['test']:
        inputs = inputs.to(device) 
        task1 = tasks[0].to(device)
        task2 = tasks[1].to(device)
        task3 = tasks[-1].to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss1 = criterion[0](outputs[0], torch.max(task1.float(), 1)[1])
            loss2 = criterion[1](outputs[1], torch.max(task2.float(), 1)[1])
            
            if task_type == 'classification':
                loss3 = criterion[2](outputs[2], torch.max(task3.float(), 1)[1])
            elif task_type == 'regression':
                loss3 = criterion[2](outputs[2].flatten(), task3.float())
            
            loss = loss1 + loss2 + loss3
        
        running_loss += loss.item() * inputs.size(0)
        task1_corrects += torch.sum(torch.max(outputs[0], 1)[1]
                                    == torch.max(task1, 1)[1])
        task2_corrects += torch.sum(torch.max(outputs[1], 1)[1] 
                                    == torch.max(task2, 1)[1])
         
        if task_type == 'classification':
            task3_corrects += torch.sum(torch.max(outputs[2], 1)[1] 
                                        == torch.max(task3, 1)[1])
        elif task_type == 'regression':
            task3_corrects += torch.sum(torch.abs(outputs[2].flatten() - task3)
                                        <= regression_threshold)
        
    epoch_loss = running_loss / dataset_sizes['test']
    task1_acc = task1_corrects.double() / dataset_sizes['test']
    task2_acc = task2_corrects.double() / dataset_sizes['test']
    task3_acc = task3_corrects.double() / dataset_sizes['test']
    
    print('[{}] Loss: {:.4f} Style Acc: {:.4f} Artist Acc: {:.4f} Date Acc: {:.4f}'.format(
        datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        epoch_loss, task1_acc, task2_acc, task3_acc))

    time_elapsed = datetime.now() - since
    print('[{}] Training complete in {:.0f}m {:.0f}s'.format(
        datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        time_elapsed.seconds // 60, time_elapsed.seconds % 60))
    
    return model, epoch_loss, task1_acc, task2_acc, task3_acc