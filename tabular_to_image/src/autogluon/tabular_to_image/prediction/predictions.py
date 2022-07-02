from pydoc import pathdirs
import matplotlib.pyplot as plt
import time
import os
import copy
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
from torchensemble import VotingClassifier
from torchensemble.fusion import FusionClassifier
from torchensemble.voting import VotingClassifier
from torchensemble.bagging import BaggingClassifier
from torchensemble.gradient_boosting import GradientBoostingClassifier
from torchensemble.snapshot_ensemble import SnapshotEnsembleClassifier
from torchensemble.soft_gradient_boosting import SoftGradientBoostingClassifier
from torchensemble.fusion import FusionClassifier
from autogluon.tabular_to_image.image_converter import Image_converter
from autogluon.tabular_to_image.models_zoo import ModelsZoo
class ImagePredictions:
    
    #image_data=Image_converter
    def __init__(self,data,lable,imageShape:int,saved_path:str,model_type:str='efficientnet-b0',pretrained:bool=True,**kwargs):
        self._validate_init_kwargs(kwargs)
                     
        self.lable=lable
        self.imageShape=imageShape
        self.saved_path=saved_path
        Image_converter_type = kwargs.pop('Image_converter_type', Image_converter)
        Image_converter_kwargs = kwargs.pop('Image_converter_kwargs', dict())
        lable = kwargs.get('lable', None)
        imageShape = kwargs.get('imageShape', None)
        saved_path = kwargs.get('saved_path', None)            
    
        self._Image_converter: Image_converter = Image_converter_type(label_column=self.lable,image_shape=self.imageShape,saved_path=self.saved_path,**Image_converter_kwargs)
        self._Image_converter_type = type(self._Image_converter)
        ##################
        ModelsZoo_type = kwargs.pop('ModelsZoo_type', ModelsZoo)
        ModelsZoo_kwargs = kwargs.pop('ModelsZoo_kwargs', dict())  
        self.model_type=model_type   
        #model_type = kwargs.get('model_type', None)
        num_classes =self._Image_converter.num_class(data)#self._Image_converter.num_class(data)
        self.pretrained = pretrained
              
        self._ModelsZoo: ModelsZoo = ModelsZoo_type(imageShape=self.imageShape ,model_type=self.model_type,
                                        num_classes=num_classes,pretrained=self.pretrained,**ModelsZoo_kwargs)
        self._ModelsZoo_type = type(self._ModelsZoo)

        
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
    
    
    """
    def train(self,dataloader, model, num_epochs=20):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Rprop(model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)

        model.train(True)
        results = []
        for epoch in range(num_epochs):
            optimizer.step()
            scheduler.step()
            model.train()

            running_loss = 0.0
            running_corrects = 0

            n = 0
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                n += len(labels)

            epoch_loss = running_loss / float(n)
            epoch_acc = running_corrects.double() / float(n)

            print(f'epoch {epoch}/{num_epochs} : {epoch_loss:.5f}, {epoch_acc:.5f}')
            results.append(EpochProgress(epoch, epoch_loss, epoch_acc.item()))
        return pd.DataFrame(results)
    """
  
    def generate_image(self,data):
        self._Image_converter.Image_Genartor(data=data)
     #.Image_Genartor(data) #_Image_converter_type.Image_Genartor(data)
    
    
    def train_model(self, num_epochs=3):
        #criterion = nn.CrossEntropyLoss() #optimizer = optim.Rprop(model.parameters(), lr=0.01) #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
        trainloader,valloader,_=Image_converter.image_tensor(self.saved_path)

        model=self.pick_model()
              
        criterion,optimizer,_=self._ModelsZoo.optimizer()
       
        use_gpu = torch.cuda.is_available()
        since = time.time()
        best_modefl_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        
        avg_loss = 0
        avg_acc = 0
        avg_loss_val = 0
        avg_acc_val = 0
        
        
        train_batches = len(trainloader)
        val_batches = len(valloader)
        
        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs))
            print('-' * 10)
            
            loss_train = 0
            loss_val = 0
            acc_train = 0
            acc_val = 0
            
            model.train(True)
            
            for i, data in enumerate(trainloader):
                if i % 100 == 0:
                    print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)
                    
                # Use half training dataset
                #if i >= train_batches / 2:
                #    break
                    
                inputs, labels = data
                
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                #loss_train += loss.data[0]
                loss_train += loss.item() * inputs.size(0)
                acc_train += torch.sum(preds == labels.data)
                
                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()
            
            print()
            # * 2 as we only used half of the dataset
            
            len_X_train_img,len_X_val_img,_=Image_converter.image_len(self.saved_path)
            avg_loss = loss_train * 2 / len_X_train_img #dataset_sizes[TRAIN]
            avg_acc = acc_train * 2 /len_X_train_img#dataset_sizes[TRAIN]
            
            model.train(False)
            model.eval()
                
            for i, data in enumerate(valloader):
                if i % 100 == 0:
                    print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)
                    
                inputs, labels = data
                
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
                else:
                    inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                
                #loss_val += loss.data[0]
                loss_val += loss.item() * inputs.size(0)
                acc_val += torch.sum(preds == labels.data)
                
                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()
            
            avg_loss_val = loss_val /len_X_val_img #dataset_sizes[VAL]
            avg_acc_val = acc_val /len_X_val_img #dataset_sizes[VAL]
            
            print()
            print("Epoch {} result: ".format(epoch))
            print("Avg loss (train): {:.4f}".format(avg_loss))
            print("Avg acc (train): {:.4f}".format(avg_acc))
            print("Avg loss (val): {:.4f}".format(avg_loss_val))
            print("Avg acc (val): {:.4f}".format(avg_acc_val))
            print('-' * 10)
            print()
            
            if avg_acc_val > best_acc:
                    best_acc = avg_acc_val
                    best_model_wts = copy.deepcopy(model.state_dict())
                
            elapsed_time = time.time() - since
            print()
            print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
            print("Best acc: {:.4f}".format(best_acc))
            
            model.load_state_dict(best_model_wts)
            return model,best_acc
    
    def eval_model(self):
        _,_,Testloader =Image_converter.image_tensor(self.saved_path)
        model=self.pick_model()        
        criterion,_,_=self._ModelsZoo.optimizer()
        use_gpu = torch.cuda.is_available()
        since = time.time()
        avg_loss = 0
        avg_acc = 0
        loss_test = 0
        acc_test = 0
        
        test_batches = len(Testloader)
        print("Evaluating model")
        print('-' * 10)
        
        for i, data in enumerate(Testloader):
            if i % 100 == 0:
                print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

            model.train(False)
            model.eval()
            inputs, labels = data

            if use_gpu:
                inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
            else:
                inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

            outputs = model(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            #loss_test += loss.data[0]
            loss_test += loss.item() * inputs.size(0)
            acc_test += torch.sum(preds == labels.data)

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        _,_,len_X_test_img=Image_converter.image_len(self.saved_path)  
        avg_loss = loss_test /len_X_test_img #dataset_sizes[TEST]
        avg_acc = acc_test /len_X_test_img#dataset_sizes[TEST]
        
        elapsed_time = time.time() - since
        print()
        print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
        print("Avg loss (test): {:.4f}".format(avg_loss))
        print("Avg acc (test): {:.4f}".format(avg_acc))
        print('-' * 10)
        return avg_acc 
    
    
    def init_train(self,model_type, num_epochs=3):
            #criterion = nn.CrossEntropyLoss() #optimizer = optim.Rprop(model.parameters(), lr=0.01) #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
        trainloader,valloader,_=Image_converter.image_tensor(self.saved_path)
                
        commonModels=['resnet18','resnet34','resnet50','resnet101','resnet152','alexnet','vgg11','vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg19','vgg19_bn',
                      'densenet121','densenet161','densenet169','densenet201''googlenet','shufflenet_v2_x0_5','shufflenet_v2_x1_0','mobilenet_v2','wide_resnet50_2',    'wide_resnet101_2','mnasnet0_5','mnasnet1_0',
                'efficientnet-b0','efficientnet-b1','efficientnet-b2','efficientnet-b3','efficientnet-b4','efficientnet-b5','efficientnet-b6','efficientnet-b7'                       
                'squeezenet1_0','squeezenet1_1'
                'resnext50_32x4d','resnext101_32x8d',
                'inception_v3','xception']
        
        if model_type in commonModels:
            model=self._ModelsZoo.create_model()
        else:
            raise AssertionError(f'Model "{model_type}" is not a valid model to specify as best! Valid models: {commonModels}')
        
        
        criterion,optimizer,_=self._ModelsZoo.optimizer()
       
        use_gpu = torch.cuda.is_available()
        since = time.time()
        best_modefl_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        
        avg_loss = 0
        avg_acc = 0
        avg_loss_val = 0
        avg_acc_val = 0
        
        
        train_batches = len(trainloader)
        val_batches = len(valloader)
        
        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs))
            print('-' * 10)
            
            loss_train = 0
            loss_val = 0
            acc_train = 0
            acc_val = 0
            
            model.train(True)
            
            for i, data in enumerate(trainloader):
                if i % 100 == 0:
                    print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)
                    
                # Use half training dataset
                #if i >= train_batches / 2:
                #    break
                    
                inputs, labels = data
                
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                #loss_train += loss.data[0]
                loss_train += loss.item() * inputs.size(0)
                acc_train += torch.sum(preds == labels.data)
                
                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()
            
            print()
            # * 2 as we only used half of the dataset
            
            len_X_train_img,len_X_val_img,_=Image_converter.image_len(self.saved_path)
            avg_loss = loss_train * 2 / len_X_train_img #dataset_sizes[TRAIN]
            avg_acc = acc_train * 2 /len_X_train_img#dataset_sizes[TRAIN]
            
            model.train(False)
            model.eval()
                
            for i, data in enumerate(valloader):
                if i % 100 == 0:
                    print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)
                    
                inputs, labels = data
                
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
                else:
                    inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                
                #loss_val += loss.data[0]
                loss_val += loss.item() * inputs.size(0)
                acc_val += torch.sum(preds == labels.data)
                
                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()
            
            avg_loss_val = loss_val /len_X_val_img #dataset_sizes[VAL]
            avg_acc_val = acc_val /len_X_val_img #dataset_sizes[VAL]
            
            print()
            print("Epoch {} result: ".format(epoch))
            print("Avg loss (train): {:.4f}".format(avg_loss))
            print("Avg acc (train): {:.4f}".format(avg_acc))
            print("Avg loss (val): {:.4f}".format(avg_loss_val))
            print("Avg acc (val): {:.4f}".format(avg_acc_val))
            print('-' * 10)
            print()
            
            if avg_acc_val > best_acc:
                    best_acc = avg_acc_val
                    best_model_wts = copy.deepcopy(model.state_dict())
                
            elapsed_time = time.time() - since
            print()
            print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
            print("Best acc: {:.4f}".format(best_acc))
            
            model.load_state_dict(best_model_wts)
            return model,best_acc
        
    def pick_model(self):  
        model_type=['resnet50','resnet101','resnet152','alexnet','vgg11','vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg19','vgg19_bn',
        'densenet121','densenet161','densenet169','densenet201''googlenet','shufflenet_v2_x0_5','shufflenet_v2_x1_0','mobilenet_v2','wide_resnet50_2',    'wide_resnet101_2','mnasnet0_5','mnasnet1_0',
         'efficientnet-b0','efficientnet-b1','efficientnet-b2','efficientnet-b3','efficientnet-b4','efficientnet-b5','efficientnet-b6','efficientnet-b7'                       
               'squeezenet1_0','squeezenet1_1'  'resnext50_32x4d','resnext101_32x8d',
                'inception_v3','xception']
        res=set()
        res2={}
        model=None
        epoch=1
        for i in range(len(model_type)):
           res=self.init_train(model_type[i], epoch)
        res2=dict([res])  
        for key,value in  res2.items():    
            if round(value.item(),2)>=0.84:
                model=key#.__class__.__name__
        return model
    
    def Ensemble(self):
        model=self.pick_model() 
        trainloader,valloader,Testloader,=Image_converter.image_tensor(self.saved_path)
        model=None
        score=[]
        init=True
        Ensemble_family={
            'LeNet@MNIST':[VotingClassifier(estimator=model,n_estimators=4,cuda=True),BaggingClassifier(estimator=model,n_estimators=4,cuda=True)],
            'LeNet@CIFAR-10':[GradientBoostingClassifier(estimator=model,n_estimators=4,cuda=True),FusionClassifier(estimator=model,n_estimators=4,cuda=True)],
            'ResNet@CIFAR-10':[VotingClassifier(estimator=model,n_estimators=4,cuda=True),SnapshotEnsembleClassifier(estimator=model,n_estimators=4,cuda=True)],
            'ResNet@CIFAR-100':[VotingClassifier(estimator=model,n_estimators=4,cuda=True),BaggingClassifier(estimator=model,n_estimators=4,cuda=True)]
            
            }
        if init :
            if self._Image_converter.len_dataset()<50000 and self.ImageShape<=50:
                for i in range(len(Ensemble_family['LeNet@MNIST'])):   
                    model=Ensemble_family['LeNet@MNIST'][i]
                    model.set_optimizer('Adam', lr=1e-3, weight_decay=5e-4)
                    criterion = nn.CrossEntropyLoss()
                    #model.set_criterion(criterion)
                    model.fit(trainloader,epochs=3,test_loader=Testloader)
                    accuracy,return_loss = model.evaluate(valloader,True)
                    score.append(accuracy)
                    best_accuracy=score[0]
                    for  i in score:                                          
                        if i>best_accuracy:
                            best_accuracy=i
                            
            init=False
                    
                
                
                
                
                
                
                        
        

    

    
       
    """
    def plot_results(df, figsize=(10, 5)):
        fig, ax1 = plt.subplots(figsize=figsize)

        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss', color='tab:red')
        ax1.plot(df['epoch'], df['loss'], color='tab:red')

        ax2 = ax1.twinx()
        ax2.set_ylabel('accuracy', color='tab:blue')
        ax2.plot(df['epoch'], df['accuracy'], color='tab:blue')

        fig.tight_layout()
    """