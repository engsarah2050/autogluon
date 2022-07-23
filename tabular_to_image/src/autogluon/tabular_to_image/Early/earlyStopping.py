import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torchvision.transforms import *
from torch.utils.data import DataLoader
import torch
import numpy as np
from collections import namedtuple
import pandas as pd
import time
import os
import copy
from efficientnet_pytorch import EfficientNet
from re import search

#from autogluon.TablarToImage import  Utils

import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
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
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        import torch 
        params_file_name=model.__class__.__name__ +'_checkpoint.pt'
        path_context, model_context, save_path=self.create_contexts(self.saved_path,params_file_name)
        
        if path_context is None:
            path_context = self.saved_path   
                     
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if model_context is not None:
            torch.save(model, (str(save_path) ))
        #torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
    
    def create_contexts(self, path_context: str,model_name:str):
        """Create and return paths to save model objects, the learner object.

        Parameters
        ----------
        path_context: str
            Top-level directory where models and trainer will be saved.
        """
        model_context = os.path.join(path_context, "models") + os.path.sep
        save_path = os.path.join(path_context, model_name)
        return path_context, model_context, save_path

    def set_contexts(self, path_context: str):
        """Update the path where model, learner, and trainer objects will be saved.
        Also see `create_contexts`."""
        self.path, self.model_context, self.save_path = self.create_contexts(path_context)     