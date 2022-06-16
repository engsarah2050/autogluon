from pickle import NONE
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
from autogluon.core.utils.loaders import load_pkl, load_str,load_compress
from autogluon.core.utils import get_cpu_count, get_gpu_count_all
from autogluon.core.utils import get_memory_size, bytes_to_mega_bytes
from autogluon.core.utils.savers import save_pkl, save_str
from autogluon.common.utils.utils import setup_outputdir
from autogluon.DeepInsight_auto.pyDeepInsight import ImageTransformer,LogScaler
from autogluon.tabular_to_image.img_sore import Store
from autogluon.core.Convertor_base.Covert import BaseImage_converter

class Image_converter:
    
    Dataset = TabularDataset
    convertor_file_name = 'conerter.pkl'
    _convortor_version_file_name = '__version__'
    
    ''' @property
    def _constructor(self):
        return  Image_converter
  
    @property
    def _constructor_sliced(self):
        return pd.Series '''
    
    def __init__(self,label_column,image_shape,saved_path,**kwargs):       
        
        '''if isinstance(data, pd.DataFrame):
            data = data         
        else:
            data = None '''
                          
        self.label_column=label_column
        self.image_shape=image_shape
        self.saved_path =Path(saved_path).expanduser() #setup_outputdir(path)
   
        self.store_type = kwargs.pop('store_type', Store)
        store_kwargs = kwargs.pop('store_kwargs', dict())
        
        self._store: Store = self.store_type(path=saved_path,low_memory=False,save_data=False,**store_kwargs)
        self._store_type = type(self._store)
        
        memoery= math.floor((get_memory_size())/1000)
        if(memoery<15):
            raise AssertionError(f'memory size  is required to be large enough , but was instead: {len(memoery)}')   	 
        
        #super().__init__(data, **kwargs)
        
   
    
 
    @property
    def savd_path(self):
        return self.saved_path
 
    @property
    def image_shape(self):
        return self.image_shape
    
    @property
    def lable_column(self):
        return self.label_column  
    
    
    
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA")
    
    @staticmethod
    def __get_dataset(data):
        if isinstance(data, TabularDataset):
            return data
        elif isinstance(data, pd.DataFrame):
            return TabularDataset(data)
        elif isinstance(data, str):
            return TabularDataset(data)
        elif isinstance(data, pd.Series):
            raise TypeError("data must be TabularDataset or pandas.DataFrame, not pandas.Series. \
                   To predict on just single example (ith row of table), use data.iloc[[i]] rather than data.iloc[i]")
        else:
            raise TypeError("data must be TabularDataset or pandas.DataFrame or str file path to data")
        
        
    def _validate_data(self, data):        
        data = self.__get_dataset(data)
        if isinstance(data, str):
            data = TabularDataset(data)
        if not isinstance(data, pd.DataFrame):
            raise AssertionError(f'data is required to be a pandas DataFrame, but was instead: {type(data)}')
        if len(set(data.columns)) < len(data.columns):
            raise ValueError("Column names are not unique, please change duplicated column names (in pandas: train_data.rename(columns={'current_name':'new_name'})")
        labelencoder = LabelEncoder()
        data[self.label_column] = labelencoder.fit_transform(data[self.label_column]) 
        categorical_columns=data.select_dtypes(exclude=['int64','float64']).columns
        CatBoostEncoder=ce.CatBoostEncoder(cols=categorical_columns)
        X = data.drop(self.label_column, axis=1)
        y = data[self.label_column]
        data3 = CatBoostEncoder.fit_transform(X, y)
        data3[self.label_column]=data.iloc[:,-1]
       	if (len(self.label_column)<=50000):
            if (self.image_shape==224): 
                data4=data3.sample(frac=.20,random_state=77)
                x1 = data4.drop(self.label_column, axis=1)
                y1= data4[self.label_column]
                X_train, X_test, y_train, y_test =train_test_split(x1,y1,test_size=0.2,random_state=23,stratify=y1)
       	        X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.25,random_state=00)
            elif (self.image_shape==256 or self.image_shape==299):
                data4=data3.sample(frac=.1,random_state=77)
                x1 = data4.drop(self.label_column, axis=1)
                y1= data4[self.label_column]
                X_train, X_test, y_train, y_test = train_test_split(x1,y1,test_size=0.2,random_state=23, stratify=y1)
                X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.25,random_state=00)  
        elif (len(self.label_column)>=50000) and (len(self.label_column)<=100000):
            self.image_shape=50
            data4=data3.sample(frac=.1,random_state=77)
            x1 = data4.drop(self.label_column, axis=1)
            y1= data4[self.label_column]
            X_train, X_test, y_train, y_test =train_test_split(x1,y1,test_size=0.2,random_state=23, stratify=y1)
            X_train, X_val, y_train, y_val = train_test_split (X_train,y_train,test_size=0.25,random_state=00)		
        
        if not isinstance(X_train, pd.DataFrame):
                raise AssertionError(
                f'train_data is required to be a pandas DataFrame, but was instead: {type(X_train)}')

        if len(set(X_train.columns)) < len(X_train.columns):
            raise ValueError(
                "Column names are not unique, please change duplicated column names (in pandas: train_data.rename(columns={'current_name':'new_name'})")

        #self._validate_unique_indices(data=train_data, name='train_data')
        if X_val is not None:
            if not isinstance(X_val, pd.DataFrame):
                raise AssertionError(f'X_val is required to be a pandas DataFrame, but was instead: {type(X_val)}')
            train_features = [column for column in X_train.columns if column != self.label_column]
            val_features = [column for column in X_val.columns if column !=self.label_column]
            if np.any(train_features != val_features):
                raise ValueError("Column names must match between training and val data")
        if X_test is not None:
            if not isinstance(X_test, pd.DataFrame):
                raise AssertionError(f'X_test is required to be a pandas DataFrame, but was instead: {type(X_test)}')
            train_features = [column for column in X_train.columns if column != self.label_column]
            test_features = [column for column in X_test.columns]
            if np.any(train_features != test_features):
                raise ValueError("Column names must match between training and test_data")
         
        return X_train,X_val,X_test,y_train , y_val,y_test
    
    def Image_Genartor(self,data):
        X_train,X_val,X_test,y_train , y_val,y_test=self._validate_data(data)
        ln = LogScaler()
        X_train_norm = ln.fit_transform(X_train)

        #@jit(target ="cuda") 
        tsne = TSNE(n_components=2, perplexity=30, metric='cosine',random_state=1701, n_jobs=-1)
        it = ImageTransformer(feature_extractor=tsne,pixels=self.image_shape, random_state=1701,n_jobs=-1)
        
               
        X_train_img = it.fit_transform(X_train_norm).astype('float32')
        #plt.figure(figsize=(5, 5))
        #_ = it.fit(X_train_norm, plot=True)
        self._store.reduce_memory_size(X_train_norm,remove_data=True,requires_save=True)
        train=self._store.save_train(X_train_img,y_train)
        self._store.reduce_memory_size(train,remove_data=True,requires_save=True)

        
        X_val_norm = ln.fit_transform(X_val)
        X_val_img = it.fit_transform(X_val_norm).astype('float32')
        
        self._store.reduce_memory_size(X_val_norm,remove_data=True,requires_save=True)
        val=self._store.save_val(X_val_img,y_val)
        self._store.reduce_memory_size(val,remove_data=True,requires_save=True)
        
        X_test_norm = ln.fit_transform(X_test)
        X_test_img = it.fit_transform(X_test_norm).astype('float32')
        
        self._store.reduce_memory_size(X_test_norm,remove_data=True,requires_save=True)
        test=self._store.save_test(X_test_img,y_test)
        self._store.reduce_memory_size(test,remove_data=True,requires_save=True)
        
        
       
    @classmethod
    def image_len(cls,path):
        train,val,test=cls.load_data(path=path)
        return len(train['X_train_img']),len(val['X_val_img']),len(test['X_test_img'])
    
    @classmethod
    def num_class(cls,path):
        _,_,test=cls.load_data(path=path)
        num_classes = np.unique(test['y_test']).size
        return num_classes
    
    
    @classmethod
    def load_data(cls,path:str, reset_paths=False):
        if not reset_paths:
            train =load_compress.load_train(path=path)
            val =load_compress.load_val(path=path)
            test =load_compress.load_test(path=path) 
            return train,val,test
        else:
            obj_train =load_compress.load_train(path=path)
            obj_val =load_compress.load_val(path=path)
            obj_test =load_compress.load_test(path=path) 
            
            obj_train.set_contexts(path)
            obj_train.reset_paths =reset_paths
            
            obj_val.set_contexts(path)
            obj_val.reset_paths = reset_paths  
            
            obj_test.set_contexts(path)
            obj_test.reset_paths = reset_paths          
            
            return obj_train,obj_val,obj_test
        
        
            
    @classmethod
    def image_tensor(cls,path:str): 
        preprocess = transforms.Compose([transforms.ToTensor()])    
        batch_size = 32
        
        le = LabelEncoder()
        #num_classes = np.unique(le.fit_transform(self.y_train)).size
        
        train,val,test=cls.load_data(path=path)
        
        X_train_tensor = torch.stack([preprocess(img) for img in train['X_train_img']])
        y_train_tensor = torch.from_numpy(le.fit_transform(train['y_train']))

        X_val_tensor = torch.stack([preprocess(img) for img in val['X_val_img']])
        y_val_tensor = torch.from_numpy(le.fit_transform(val['y_val'] ))

        X_test_tensor = torch.stack([preprocess(img) for img in test['X_test_img']])
        y_test_tensor = torch.from_numpy(le.transform(test['y_test']))
        
        trainset = TensorDataset(X_train_tensor, y_train_tensor)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        valset = TensorDataset(X_val_tensor, y_val_tensor)
        valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)

        Testset = TensorDataset(X_test_tensor, y_test_tensor)
        Testloader = DataLoader(Testset, batch_size=batch_size, shuffle=True)
        
        
        return trainloader,valloader,Testloader
    

    
    @classmethod
    def _load_version_file(cls, path) -> str:
        version_file_path = path + cls._convortor_version_file_name 
        version = load_str.load(path=version_file_path)
        return version

    def _save_version_file(self, silent=False):
        from ..version import __version__
        version_file_contents = f'{__version__}'
        version_file_path = self.path + self._convortor_version_file_name 
        save_str.save(path=version_file_path, data=version_file_contents, verbose=not silent)

    