from nis import cat
from pydoc import pathdirs
from warnings import catch_warnings
import matplotlib.pyplot as plt
import time
import os
import copy
import logging
import torch
#device = torch.device("cuda") #device = 'cuda'
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from autogluon.core.utils import get_memory_size, bytes_to_mega_bytes
from autogluon.core.models.abstract.abstract_nn_model import AbstractNeuralNetworkModel
from autogluon.core.utils import try_import_torch,try_import_torchensemble
from autogluon.core.utils.loaders import load_compress
from autogluon.tabular_to_image.image_converter import Image_converter
from autogluon.tabular_to_image.models_zoo import ModelsZoo
from autogluon.tabular_to_image.Early.earlyStopping import EarlyStopping


__all__ = ['ImagePredictor']


logger = logging.getLogger(__name__)  # return autogluon root logger

class ImagePredictions:#(AbstractNeuralNetworkModel):
    
        
    #image_data=Image_converter
    def __init__(self,data,lable,imageShape,saved_path:str,model_type:str,patience,delta:int=0,pretrained:bool=True,**kwargs):
        try_import_torch()
        #super().__init__(**kwargs)
        self._validate_init_kwargs(kwargs)           
        self.lable=lable
        self.imageShape=imageShape
        self.saved_path=saved_path
        self.is_data_saved=False
        self.patience=patience
        self.delta=delta, 
        #####################################
        Image_converter_type = kwargs.pop('Image_converter_type', Image_converter)
        Image_converter_kwargs = kwargs.pop('', dict())
        lable = kwargs.get('lable', None)
        imageShape = kwargs.get('imageShape', None)
        saved_path = kwargs.get('saved_path', None)            
    
        self._Image_converter: Image_converter = Image_converter_type(label_column=self.lable,image_shape=self.imageShape,saved_path=self.saved_path,**Image_converter_kwargs)
        self._Image_converter_type = type(self._Image_converter)
        ##################Image_converter_kwargs
        ModelsZoo_type = kwargs.pop('ModelsZoo_type', ModelsZoo)
        ModelsZoo_kwargs = kwargs.pop('ModelsZoo_kwargs', dict())  
        self.model_type=model_type   
        #model_type = kwargs.get('model_type', None)
        num_classes =self._Image_converter.num_class(data)#self._Image_converter.num_class(data)
        self.pretrained = pretrained
              
        self._ModelsZoo: ModelsZoo = ModelsZoo_type(imageShape=self.imageShape ,model_type=self.model_type,
                                        num_classes=num_classes,pretrained=self.pretrained,**ModelsZoo_kwargs)
        self._ModelsZoo_type = type(self._ModelsZoo)
        #####################################
        EarlyStopping_type = kwargs.pop('EarlyStopping_type', EarlyStopping)
        EarlyStopping_kwargs = kwargs.pop('EarlyStopping_kwargs', dict()) 
        
        self._EarlyStopping: EarlyStopping = EarlyStopping_type(patience=self.patience, verbose=False, delta=self.delta, saved_path=self.saved_path, trace_func=print,**EarlyStopping_kwargs)
        self._EarlyStopping_type = type(self._EarlyStopping)

        
    @property
    def Label_column(self): 
        return self._Image_converter.label_column
    @property
    def ImageShape(self):
        return self._Image_converter.image_shape
    @property
    def Model_type(self):
        return self._ModelsZoo.model_type
    ''' @property
    def Num_classes(self):
        return  self._Image_converter.num_class(data) '''
    @property
    def Pretrained(self):
        return self._ModelsZoo.pretrained
    @property
    def Model(self):
        return self._ModelsZoo.create_model() 
     
    
    @staticmethod
    def _validate_init_kwargs(kwargs):
        valid_kwargs = {
            'Image_converter_type',
            'Image_converter_kwargs',
            'lable',
            'saved_path',
            'ModelsZoo_type',
            'ModelsZoo_kwargs',
            'imageShape',
            'model_type',
            #'num_classes',
            'pretrained',
        }
        invalid_keys = []
        for key in kwargs:
            if key not in valid_kwargs:
                invalid_keys.append(key)
        if invalid_keys:
            raise ValueError(f'Invalid kwargs passed: {invalid_keys}\nValid kwargs: {list(valid_kwargs)}') 
       
    def generate_image(self,data):
        self._Image_converter.Image_Genartor(data=data)
     #.Image_Genartor(data) #_Image_converter_type.Image_Genartor(data)
                       
        
    def init_train(self,model_type, epochs,patience,scheduler=None):
        #criterion = nn.CrossEntropyLoss() #optimizer = optim.Rprop(model.parameters(), lr=0.01) #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
        trainloader,valloader,_=Image_converter.image_tensor(self.saved_path)
                
        commonModels=[#'resnet18','resnet34','resnet50','resnet101','resnet152', 
                      #'regnet_x_16gf',regnet_x_1_6gf,'regnet_x_32gf','regnet_x_3_2gf','regnet_x_400mf','regnet_x_800mf','regnet_x_8gf',
                      #'regnet_y_128gf','regnet_y_16gf','regnet_y_1_6gf','regnet_y_32gf','regnet_y_3_2gf','regnet_y_400mf','regnet_y_800mf',
                      'densenet121','densenet161','densenet169','densenet201'#,
                    #  'alexnet','vgg11','vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg19','vgg19_bn',
                    #  'googlenet','shufflenet_v2_x0_5','shufflenet_v2_x1_0','mobilenet_v2','wide_resnet50_2', 'wide_resnet101_2','mnasnet0_5','mnasnet1_0',
                    #  'efficientnet-b0','efficientnet-b1','efficientnet-b2','efficientnet-b3','efficientnet-b4','efficientnet-b5','efficientnet-b6','efficientnet-b7' ,
                    #  'efficientnet_v2_s','efficientnet_v2_m','efficientnet_v2_l','convnext_small','convnext_base','convnext_large',  
                    #'swin_t','swin_s','swin_b',                 
                    #  'squeezenet1_0','squeezenet1_1','resnext50_32x4d','resnext101_32x8d','inception_v3','xception'
                    ]
        
        if model_type in commonModels:
            model=self._ModelsZoo.create_model()
        else:
            raise AssertionError(f'Model "{model_type}" is not a valid model to specify as best! Valid models: {commonModels}')      
        criterion,optimizer,_=self._ModelsZoo.optimizer(model)
        """
        Helper function for train model
        :param model: current model
        :param train_loader: train data loader
        :param test_loader: test data loader
        :param epochs: number of epoch
        :param optimizer: optimizer
        :param criterion: loss function
        :param scheduler: scheduler, default None
        :param name: model name, default model.pt
        :param path: model saved location, default None
        :return: model, list of train loss and test loss
        """

        # compare overfitted
        train_loss_data, valid_loss_data = [], []
        # check for validation loss
        valid_loss_min = np.Inf
        # calculate time
        since = time.time()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # initialize the early_stopping object
        early_stopping =self._EarlyStopping_type(patience,self.saved_path,verbose=True )

        for epoch in range(epochs):
            print("Epoch: {}/{}".format(epoch+1 , epochs))
            # monitor training loss
            train_loss = 0.0
            valid_loss = 0.0
            total = 0
            correct = 0
            Accuracy=0.0
            e_since = time.time()
            model.train()  # prep model for training

            #train
            for images, labels in trainloader:
                # Move input and label tensors to the default device
                images, labels = images.to(device), labels.to(device)
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                log_ps = model(images)
                # calculate the loss
                loss = criterion(log_ps, labels)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update running training loss
                train_loss += loss.item() * images.size(0)
                #print("\t\tGoing for validation")
                model.eval()  # prep model for evaluation
            #validate
            for data, target in valloader:
                # Move input and label tensors to the default device
                data, target = data.to(device), target.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the loss
                loss_p = criterion(output, target)
                # update running validation loss
                valid_loss += loss_p.item() * data.size(0)
                # calculate accuracy
                proba = torch.exp(output)
                top_p, top_class = proba.topk(1, dim=1)
                equals = top_class == target.view(*top_class.shape)
                # accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            # print training/validation statistics
            # calculate average loss over an epoch
            train_loss = train_loss / len(trainloader.dataset)
            valid_loss = valid_loss / len(valloader.dataset)

            # calculate train loss and running loss
            train_loss_data.append(train_loss * 100)
            valid_loss_data.append(valid_loss * 100)
            Accuracy=correct / total * 100

            print("\tTrain loss:{:.6f}..".format(train_loss),
                "\tValid Loss:{:.6f}..".format(valid_loss),
                "\tAccuracy: {:.4f}".format(correct / total * 100))

            if scheduler is not None:
                scheduler.step()  # step up scheduler
            
            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('\tValidation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
                #torch.save(model.state_dict(), name)
                valid_loss_min = valid_loss
                # save to google drive
                #if path is not None:
                #    torch.save(model.state_dict(), path)

            # Time take for one epoch
            time_elapsed = time.time() - e_since
            print('\tEpoch:{} completed in {:.0f}m {:.0f}s'.format(
                epoch + 1, time_elapsed // 60, time_elapsed % 60))

        # compare total time
        time_elapsed = time.time() - since
        print('Training completed in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        # load best model
        #model = load_latest_model(model, name)
        self.reduce_memory_size(trainloader)
        self.reduce_memory_size(valloader)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
        # return the model
        return model, [Accuracy,train_loss_data, valid_loss_data]       

    
    def train_model(self,model,  patience, n_epochs):
        trainloader,valloader,_=Image_converter.image_tensor(self.saved_path)
        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = [] 
        
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        criterion,optimizer,_=self._ModelsZoo.optimizer(model)
        for epoch in range(1, n_epochs + 1):

            ###################
            # train the model #
            ###################
            model.train() # prep model for training
            for batch, (data, target) in enumerate(trainloader, 1):
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the loss
                loss = criterion(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # record training loss
                train_losses.append(loss.item())

            ######################    
            # validate the model #
            ######################
            model.eval() # prep model for evaluation
            for data, target in valloader:
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the loss
                loss = criterion(output, target)
                # record validation loss
                valid_losses.append(loss.item())

            # print training/validation statistics 
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            
            epoch_len = len(str(n_epochs))
            
            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                        f'train_loss: {train_loss:.5f} ' +
                        f'valid_loss: {valid_loss:.5f}')
            
            print(print_msg)
            
            # clear lists to track next epoch
            train_losses = []
            valid_losses = []
            
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
        self.reduce_memory_size(trainloader)
        self.reduce_memory_size(valloader)    
        # load the last checkpoint with the best model
        #model.load_state_dict(torch.load('checkpoint.pt'))

        return  model, avg_train_losses, avg_valid_losses
        
    def traindata(self,model_type, epochs):
        #criterion = nn.CrossEntropyLoss() #optimizer = optim.Rprop(model.parameters(), lr=0.01) #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
        trainloader,valloader,Testloader=Image_converter.image_tensor(self.saved_path)
                
        commonModels=[#'resnet18','resnet34','resnet50','resnet101','resnet152', 
                      #'regnet_x_16gf',regnet_x_1_6gf,'regnet_x_32gf','regnet_x_3_2gf','regnet_x_400mf','regnet_x_800mf','regnet_x_8gf',
                      #'regnet_y_128gf','regnet_y_16gf','regnet_y_1_6gf','regnet_y_32gf','regnet_y_3_2gf','regnet_y_400mf','regnet_y_800mf',
                      'densenet121'#,'densenet161','densenet169','densenet201',
                    #  'alexnet','vgg11','vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg19','vgg19_bn',
                    #  'googlenet','shufflenet_v2_x0_5','shufflenet_v2_x1_0','mobilenet_v2','wide_resnet50_2', 'wide_resnet101_2','mnasnet0_5','mnasnet1_0',
                    #  'efficientnet-b0','efficientnet-b1','efficientnet-b2','efficientnet-b3','efficientnet-b4','efficientnet-b5','efficientnet-b6','efficientnet-b7' ,                      
                    #  'efficientnet_v2_s','efficientnet_v2_m','efficientnet_v2_l','convnext_small','convnext_base','convnext_large',
                    #'swin_t','swin_s','swin_b', 
                    #  'squeezenet1_0','squeezenet1_1','resnext50_32x4d','resnext101_32x8d','inception_v3','xception'
                    ]
        
        if model_type in commonModels:
            model=self._ModelsZoo.create_model()
        else:
            raise AssertionError(f'Model "{model_type}" is not a valid model to specify as best! Valid models: {commonModels}')
        
        
        criterion,optimizer,_=self._ModelsZoo.optimizer(model)
        
        # Early stopping
        last_loss = 100
        patience = 2
        triggertimes = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for epoch in range(1, epochs+1):
            model.train()

            for times, data in enumerate(trainloader, 1):
                input = data[0].to(device)
                label = data[1].to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward and backward propagation
                output = model(input.view(input.shape[0], -1))
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                # Show progress
                if times % 100 == 0 or times == len(trainloader):
                    print('[{}/{}, {}/{}] loss: {:.8}'.format(epoch, epochs, times, len(trainloader), loss.item()))

            # Early stopping       
            # 
        self.reduce_memory_size(trainloader)
        model.train(False)
        model.eval()
        loss_total = 0

        # Test validation data
        with torch.no_grad():
            for data in valloader:
                input = data[0].to(device)
                label = data[1].to(device)

                output = model(input.view(input.shape[0], -1))
                loss = criterion(output, label)
                loss_total += loss.item()

        current_loss= loss_total / len(valloader)
            
        print('The Current Loss:', current_loss)

        if current_loss > last_loss:
            trigger_times += 1
            print('Trigger Times:', trigger_times)
            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                return model

            else:
                print('trigger times: 0')
                trigger_times = 0

            last_loss = current_loss
        self.reduce_memory_size(valloader)           
        #################
        #   test        #    
        #################
        total = 0
        correct = 0
        Accuracy=0.0 
        with torch.no_grad():
            for data in Testloader:
                input = data[0].to(device)
                label = data[1].to(device)

                output = model(input.view(input.shape[0], -1))
                _, predicted = torch.max(output.data, 1)

                total += label.size(0)
                correct += (predicted == label).sum().item()

        #print('Accuracy:', correct / total) 
        Accuracy=correct / total
        self.reduce_memory_size(Testloader)  
        return model,Accuracy#,last_loss 
        
    def pick_model(self):  
        model_type=[#'resnet50','resnet101','resnet152',
                    #'regnet_x_16gf',regnet_x_1_6gf,'regnet_x_32gf','regnet_x_3_2gf','regnet_x_400mf','regnet_x_800mf','regnet_x_8gf',
                    #'regnet_y_128gf','regnet_y_16gf','regnet_y_1_6gf','regnet_y_32gf','regnet_y_3_2gf','regnet_y_400mf','regnet_y_800mf',                    
                    'densenet121','densenet161','densenet169','densenet201'#,
               #     'alexnet' ,'vgg11','vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg19','vgg19_bn',
               #     'googlenet','shufflenet_v2_x0_5','shufflenet_v2_x1_0','mobilenet_v2','wide_resnet50_2',    'wide_resnet101_2','mnasnet0_5','mnasnet1_0',
               #     'efficientnet-b0','efficientnet-b1','efficientnet-b2','efficientnet-b3','efficientnet-b4','efficientnet-b5','efficientnet-b6','efficientnet-b7',                       
               #      'efficientnet_v2_s','efficientnet_v2_m','efficientnet_v2_l','convnext_small','convnext_base','convnext_large',
               #'swin_t','swin_s','swin_b', 
               #     'squeezenet1_0','squeezenet1_1' , 'resnext50_32x4d','resnext101_32x8d',
               #     'inception_v3','xception'
                ]
        res=set()
        res2={}
        model=None
        epoch=10
        patience=3
        for i in range(len(model_type)):
           k,v=self.init_train(model_type[i], epoch,patience)
        res2[k]=v
        #res2=dict([res])  
        for key,value in  res2.items():    
            if round(value[0],2)>=0.80:
                model=key#.__class__.__name__      

        savepath=self.save_model(model)
        if savepath is not None:
            self.reduce_memory_size(res)
            self.reduce_memory_size(res2)
        else:
            raise AssertionError(f'Model "{model}" is not saved') 
        
        model=ImagePredictions.load(savepath)           
        return model   #,savepath
    
    def save_model(self,model, verbose=True) -> str:
        import torch 
        params_file_name=model.__class__.__name__ +".pt"
        path_context, model_context, save_path=self.create_contexts(self.saved_path,params_file_name)
        
        if path_context is None:
            path_context = self.saved_path   
                     
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if model_context is not None:
            torch.save(model, (str(save_path) ))   
        if verbose: logger.log(15, 'Loading: %s' % save_path)         
        self.is_data_saved=True
        if save_path is not None and self.is_data_saved:
            self.reduce_memory_size(model)
            torch.cuda.empty_cache()
        
        
        return save_path

    @classmethod
    def load(cls,path: str, reset_paths=False,verbose=True):
        import torch
        obj: ModelsZoo = load_compress.load_model(path,verbose=True)
        #load_pkl.load(path=path + cls.model_file_name, verbose=verbose)
        if reset_paths:
            obj.set_contexts(path)
        
        #obj.model = load_compress.load_model(path)
        return obj
    
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

    
    def reduce_memory_size(self, data_files, remove_data=True, requires_save=True):
        if remove_data or self.is_data_saved:
            try:
                del data_files
            except NameError:
                print('f Variable {data_files} is not defined')
            except OSError:
                pass
        if requires_save:
                self.is_data_saved = False   
    
    def single_model(self):
        model=self.pick_model()
        epoch=10
        patience=3 
        model2,avg_train_losses, avg_valid_losses=self.train_model(model,patience, epoch)
        path=self.save_model(model2, verbose=True)
        model3=ImagePredictions.load(path, reset_paths=False,verbose=True)
        return model3,avg_train_losses, avg_valid_losses
        
    def init_Ensemble(self,model):
        try_import_torchensemble()
        from torchensemble.fusion import FusionClassifier
        from torchensemble.voting import VotingClassifier
        from torchensemble.bagging import BaggingClassifier
        from torchensemble.gradient_boosting import GradientBoostingClassifier
        from torchensemble.snapshot_ensemble import SnapshotEnsembleClassifier
        from torchensemble.soft_gradient_boosting import SoftGradientBoostingClassifier
        
        
        trainloader,valloader,Testloader=Image_converter.image_tensor(self.saved_path)     
        init_model=None
        score=[]
        lose=0.0
        return_losses=[]
        #epochs=3
        initmodels={}
        Ensemble_family={
                'models':[FusionClassifier,VotingClassifier,BaggingClassifier,GradientBoostingClassifier,SnapshotEnsembleClassifier,SoftGradientBoostingClassifier],

                    }
        family='LeNet'
        epochs=1#correct number is  5 and so do estimator or its multipls
        lr=1e-3
        maxvalue=[0.0]
        optm='Adam'
        n_estimators=2
        maximum=['']
        tem=0.0
        tem_est=' '
        familes=[[{'name':'LeNet_family'},{"optm":'Adam'},{"lr":1e-3},{"n_estimators":2},{"epochs":10}],
                 [{"name":'ResNet_family'},{"optm":'SGD'},{"lr":1e-1},{"n_estimators":5},{"epochs":10}]
                                    
                ]
        i=1
        res={}
        for t in range(len(familes)) :
            for l in range(len(Ensemble_family['models'])): 
                init_model=Ensemble_family['models'][l](estimator=model,n_estimators=n_estimators,cuda=True)
                init_model.set_optimizer(optm, lr=lr, weight_decay=5e-4)
                criterion = nn.CrossEntropyLoss()
                init_model.set_criterion(criterion)
                init_model.fit(trainloader,epochs=epochs,test_loader=Testloader)
                accuracy,return_loss = init_model.evaluate(Testloader,True)
                score.append(accuracy)
                return_losses.append(return_loss)             
                initmodels[init_model._get_name()]=[accuracy,return_loss]
                #initmodels[init_model._get_name()]=return_loss
                best_accuracy=score[0]
                lose=return_losses[0]
                #del  trainloader
                #del  Testloader
                for j in score:                                          
                    if j>best_accuracy:
                        best_accuracy=j
                maxvalue.append((
                    (max(initmodels, key=initmodels.get),initmodels[max(initmodels, key=initmodels.get)]),(f'no.epoch{epochs}',family)))
                maximum.append(f'no.group{t}' ) 
                i=i+1
                #estimator=2
                optm='SGD'
                lr=1e-1
                #epochs=2
                score.clear()
                tem=maxvalue 
                tem_est=maximum
                family='ResNet' 
                

        #val=unzip(*maxvalue)
        import itertools
        b=list(itertools.chain(*res['group1']))
        res = dict(zip(maximum, b))
        import itertools
        #type(res.items())
        #val0=list(itertools.chain(*res['no.group0']))
        #val1=list(itertools.chain(*res['no.group1']))
        res3={}
        res3['LeNet_family']=list(res['no.group0'])
        res3['ResNet_family']=list(res['no.group1'])
        maz=res3['LeNet_family'][1][0]
        maz_lose=maz=res3['LeNet_family'][1][1]
        if maz<res3['ResNet_family'][1][0]:
            maz=res3['ResNet_family'][1][0]
        elif maz==res3['ResNet_family'][1][0] :
            if maz_lose>res3['ResNet_family'][1][1]:
                maz_lose=res3['ResNet_family'][1][1]     
        key=([k for k,v in res3.items() if v[1][0] == maz]) 
        for i in range(len(familes.items())):
            if key[0]==(list(familes[i][0].values())[0]):
                return familes[i]
        self.reduce_memory_size(init_model)
        self.reduce_memory_size(trainloader)
        self.reduce_memory_size(valloader)
        self.reduce_memory_size(Testloader)         
        return res3[key[0]][0]
        #torch.cuda.empty_cache()
    def train_ensamble(self,ensmble_model,family,model) :
        try_import_torchensemble()
        from torchensemble.fusion import FusionClassifier
        from torchensemble.voting import VotingClassifier
        from torchensemble.bagging import BaggingClassifier
        from torchensemble.gradient_boosting import GradientBoostingClassifier
        from torchensemble.snapshot_ensemble import SnapshotEnsembleClassifier
        from torchensemble.soft_gradient_boosting import SoftGradientBoostingClassifier
        
        
        trainloader,valloader,Testloader,=Image_converter.image_tensor(self.saved_path)     
        init_model=None
        #epochs=3
        Ensemble_family={
                'models':[FusionClassifier,VotingClassifier,BaggingClassifier,GradientBoostingClassifier,SnapshotEnsembleClassifier,SoftGradientBoostingClassifier],
        }
        ensamble_name=eval(ensmble_model)
        if ensamble_name in Ensemble_family:
            init_model=ensamble_name(estimator=model,n_estimators=family[3]["n_estimators"],cuda=True)
            init_model.set_optimizer(family[1]["optm"], lr=family[2]["lr"], weight_decay=5e-4)
            criterion = nn.CrossEntropyLoss()
            init_model.set_criterion(criterion)
            init_model.fit(trainloader,epochs=one_familes[4]["epochs"],test_loader=Testloader)
            accuracy,return_loss = init_model.evaluate(Testloader,True)
            return accuracy
        else:
            raise AssertionError(f'Model "{ensamble_name}" is not a valid model to specify as best! Valid models: {Ensemble_family}')
        
    def final_ensamble(self):
        model=self.single_model()
        ensamble_model,family=self.init_Ensemble(model) 
        accuracy=self.train_ensamble(ensamble_model,family,model)  
        return  accuracy
        
        
           