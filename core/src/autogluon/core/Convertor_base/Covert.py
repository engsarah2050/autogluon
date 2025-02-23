from re import T
import matplotlib.pyplot as plt
import time
import os
import math
import copy
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.ticker as ticker
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
#device = torch.device("cuda") #device = 'cuda'
import torchvision.transforms as transforms
import torchvision
from torchvision import datasets, models, transforms

from autogluon.core.dataset import TabularDataset
from autogluon.core.utils.loaders import load_pkl, load_str
from autogluon.core.utils.utils import ResourceManager#get_cpu_count, get_gpu_count_all
#from autogluon.core.utils import ResourceManager# get_memory_size, bytes_to_mega_bytes
from autogluon.core.utils.savers import save_pkl, save_str
from autogluon.common.utils.utils import setup_outputdir
from autogluon.DeepInsight_auto.pyDeepInsight import ImageTransformer,LogScaler
from autogluon.tabular_to_image.img_sore import Store

class BaseImage_converter(pd.DataFrame):
    
    @property
    def _constructor(self):
        return  BaseImage_converter
  
    @property
    def _constructor_sliced(self):
        return pd.Series
  
    #_metadata = ['label_column,image_shape']  
    def __init__(self,data,**kwargs): 
        if isinstance(data, pd.DataFrame):
            data = data         
        else:
            data = None
        
        super().__init__(data, **kwargs)
         
        
  
  