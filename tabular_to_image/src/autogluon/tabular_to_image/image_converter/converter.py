import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from autogluon.common.utils.utils import setup_outputdir
import torch
#device = torch.device("cuda") #device = 'cuda'
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from autogluon.core.dataset import TabularDataset
from autogluon.core.utils.loaders import load_pkl, load_str
from autogluon.core.utils.savers import save_pkl, save_str
from autogluon.DeepInsight_auto.pyDeepInsight import ImageTransformer,LogScaler

from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
#from sklearn.model_selection import StratifiedKFol
from sklearn.manifold import TSNE
class Image_converter:
    
    Dataset = TabularDataset
    convertor_file_name = 'conerter.pkl'
    _convortor_version_file_name = '__version__'
    
    def __init__(self, label_column,image_shape,path=None):
      #self.train_dataset=train_dataset
      self.label_column=label_column
      self.image_shape=image_shape
      path = setup_outputdir(path)
     
    #def data_split(self,):
    #    X_train, X_test, y_train, y_test = train_test_split(self.train_dataset,  self.label_column, test_size=0.2)
    #    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25
   
   
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
        
        X_train, X_test, y_train, y_test = train_test_split(data,  self.label_column, test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)
        
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
            train_features = [column for column in X_val.columns if column != y_train]
            val_features = [column for column in X_val.columns if column != y_val]
            if np.any(train_features != val_features):
                raise ValueError("Column names must match between training and val data")
        if X_test is not None:
            if not isinstance(X_test, pd.DataFrame):
                raise AssertionError(f'X_test is required to be a pandas DataFrame, but was instead: {type(X_test)}')
            train_features = [column for column in X_train.columns if column != y_train]
            test_features = [column for column in X_test.columns]
            if np.any(train_features != test_features):
                raise ValueError("Column names must match between training and test_data")
         
        return X_train,X_val,X_test,y_train , y_val,y_test    
     
    def Image_Genartor(self,data):
        X_train,X_val,X_test,_ , _,_=self._validate_data(data)
        ln = LogScaler()
        X_train_norm = ln.fit_transform(X_train)
        X_val_norm = ln.fit_transform(X_val)
        X_test_norm = ln.transform(X_test)
        #@jit(target ="cuda") 
        it = ImageTransformer(feature_extractor='tsne',pixels=self.image_shape, random_state=1701,n_jobs=-1)
       
        X_train_img = it.fit_transform(X_train_norm)
        X_val_img = it.fit_transform(X_val_norm)
        X_test_img = it.transform(X_test_norm)

        tsne = TSNE(n_components=2, perplexity=30, metric='cosine',random_state=1701, n_jobs=-1)

        plt.figure(figsize=(5, 5))
        _ = it.fit(X_train_norm, plot=True)
        return X_train_img,X_val_img,X_test_img
    
    def image_len(self,data):
        X_train_img,X_val_img,X_test_img=self.Image_Genartor(data)
        return len(X_train_img),len(X_val_img),len(X_test_img)
            
    def image_tensor(self,data): 
        
        preprocess = transforms.Compose([transforms.ToTensor()])    
        batch_size = 64
        
        le = LabelEncoder()
        #num_classes = np.unique(le.fit_transform(self.y_train)).size
        _,_,_,y_train , y_val,y_test=self._validate_data(data)
        X_train_img,X_val_img,X_test_img=self.Image_Genartor(self.image_shape)
        X_train_tensor = torch.stack([preprocess(img) for img in X_train_img ])
        y_train_tensor = torch.from_numpy(le.fit_transform(y_train))

        X_val_tensor = torch.stack([preprocess(img) for img in X_val_img])
        y_val_tensor = torch.from_numpy(le.fit_transform(y_val ))

        X_test_tensor = torch.stack([preprocess(img) for img in X_test_img])
        y_test_tensor = torch.from_numpy(le.transform(y_test))
        
        trainset = TensorDataset(X_train_tensor, y_train_tensor)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

        valset = TensorDataset(X_val_tensor, y_val_tensor)
        valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)

        Testset = TensorDataset(X_test_tensor, y_test_tensor)
        Testloader = DataLoader(Testset, batch_size=batch_size, shuffle=True)
        return trainloader,valloader,Testloader#,num_classes 
    
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

    def save(self, silent=False):
        """
        Save this Predictor to file in directory specified by this Predictor's `path`.
        Note that :meth:`TabularPredictor.fit` already saves the predictor object automatically
        (we do not recommend modifying the Predictor object yourself as it tracks many trained models).

        Parameters
        ----------
        silent : bool, default = False
            Whether to save without logging a message.
        """
        path = self.path
        tmp_learner = self._learner
        tmp_trainer = self._trainer
        self._learner.save()
        self._learner = None
        self._trainer = None
        save_pkl.save(path=path + self.convertor_file_name, object=self)
        self._learner = tmp_learner
        self._trainer = tmp_trainer
        self._save_version_file(silent=silent)
        if not silent:
            logger.log(20, f'images saved. To load, use: convertor = Image_converter.load("{self.path}")')

    @classmethod
    def _load(cls, path: str):
        """
        Inner load method, called in `load`.
        """
        convertor: Image_converter = load_pkl.load(path=path + cls.convertor_file_name)
        learner = convertor._learner_type.load(path)
        convertor._set_post_fit_vars(learner=learner)
        return convertor

    @classmethod
    def load(cls, path: str, verbosity: int = None, require_version_match: bool = True):
        """
        Load a TabularPredictor object previously produced by `fit()` from file and returns this object. It is highly recommended the predictor be loaded with the exact AutoGluon version it was fit with.

        Parameters
        ----------
        path : str
            The path to directory in which this Predictor was previously saved.
        verbosity : int, default = None
            Sets the verbosity level of this Predictor after it is loaded.
            Valid values range from 0 (least verbose) to 4 (most verbose).
            If None, logging verbosity is not changed from existing values.
            Specify larger values to see more information printed when using Predictor during inference, smaller values to see less information.
            Refer to TabularPredictor init for more information.
        require_version_match : bool, default = True
            If True, will raise an AssertionError if the `autogluon.tabular` version of the loaded predictor does not match the installed version of `autogluon.tabular`.
            If False, will allow loading of models trained on incompatible versions, but is NOT recommended. Users may run into numerous issues if attempting this.
        """
        if verbosity is not None:
            set_logger_verbosity(verbosity)  # Reset logging after load (may be in new Python session)
        if path is None:
            raise ValueError("path cannot be None in load()")

        try:
            from ..version import __version__
            version_load = __version__
        except:
            version_load = None

        path = setup_outputdir(path, warn_if_exist=False)  # replace ~ with absolute path if it exists
        try:
            version_init = cls._load_version_file(path=path)
        except:
            logger.warning(f'WARNING: Could not find version file at "{path + cls._predictor_version_file_name}".\n'
                           f'This means that the predictor was fit in a version `<=0.3.1`.')
            version_init = None

        if version_init is None:
            predictor = cls._load(path=path)
            try:
                version_init = predictor._learner.version
            except:
                version_init = None
        else:
            predictor = None
        if version_init is None:
            version_init = 'Unknown (Likely <=0.0.11)'
        if version_load != version_init:
            logger.warning('')
            logger.warning('############################## WARNING ##############################')
            logger.warning('WARNING: AutoGluon version differs from the version used to create the predictor! '
                           'This may lead to instability and it is highly recommended the predictor be loaded '
                           'with the exact AutoGluon version it was created with.')
            logger.warning(f'\tPredictor Version: {version_init}')
            logger.warning(f'\tCurrent Version:   {version_load}')
            logger.warning('############################## WARNING ##############################')
            logger.warning('')
            if require_version_match:
                raise AssertionError(
                    f'Predictor was created on version {version_init} but is being loaded with version {version_load}. '
                    f'Please ensure the versions match to avoid instability. While it is NOT recommended, '
                    f'this error can be bypassed by specifying `require_version_match=False`.')

        if predictor is None:
            predictor = cls._load(path=path)

        return predictor    