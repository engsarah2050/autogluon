from typing_extensions import IntVar
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

class ModelsZoo():  
    commonShapes=[224,227,256,299]
    
    def __init__(self,imageShape,model_type, num_classes, pretrained):  
        self.imageShape = imageShape 
        self.model_type=model_type
        self.num_classes=num_classes
        self.pretrained=pretrained
        
        #use_gpu = torch.cuda.is_available() 
         
    
    @property
    def ImageShape(self)-> int:
        return int(self.imageShape)
 
    @property
    def MODEL(self):
        return self.model_type
    
    @property
    def N_class(self):
        return self.num_classes
        
    @property
    def Pretrain(self):
        return self.pretrained
    
    def create_model(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        models_list=['resnet','alexnet','vgg','densenet','googlenet','shufflenet',
                     'mobilenet','wide_resnet','efficientnet','squeezenet',
                     'mnasnet','resnext','inception']
        x=[i for i in models_list if i in self.model_type]
        model = None
        if int(self.ImageShape)==self.commonShapes[0]:
            if x[0]== 'resnet':
                if self.model_type =='resnet18':
                    model = models.resnet18(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = False
                    model.fc = nn.Linear(model.fc.in_features,self.num_classes )
                elif  self.model_type=='resnet34'  :
                    model = models.resnet34(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = False
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
                elif self.model_type== 'resnet50' :
                    model = models.resnet50(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = False
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
                elif self.model_type=='resnet101'  :
                    model = models.resnet101(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = False
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
                elif self.model_type== 'resnet152' :
                    model = models.resnet152(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = False
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            if x[0]=='alexnet':
                model = models.alexnet(pretrained=self.pretrained).to(device)
                for param in model.parameters():
                    param.requires_grad = False
                model.classifier[6] = nn.Linear(4096, self.num_classes)
            if x[0]== 'vgg' :
                if self.model_type=='vgg11' :
                    model = models.vgg11(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = False
                    model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)
                elif self.model_type =='vgg11_bn' :
                    model = models.vgg11_bn(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = False
                    model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)
                elif self.model_type== 'vgg13' :
                    model = models.vgg13(pretrained=self.pretrained).to(device).double()
                    for param in model.parameters():
                        param.requires_grad = False    
                    model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)
                elif self.model_type == 'vgg13_bn' :
                    model = models.vgg13_bn(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = False 
                    model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)
                elif self.model_type == 'vgg16' :
                    model = models.vgg16(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = False 
                    model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)
                elif self.model_type== 'vgg16_bn' :
                    model = models.vgg16_bn(pretrained=self.pretrained).to(device).double()
                    for param in model.parameters():
                        param.requires_grad = False 
                    model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)
                elif self.model_type=='vgg19' :
                    model = models.vgg19(pretrained=self.pretrained).to(device).double()
                    for param in model.parameters():
                        param.requires_grad = False 
                    model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)
                elif self.model_type=='vgg19_bn':
                    model = models.vgg19_bn(pretrained=self.pretrained).to(device).double()
                    for param in model.parameters():
                        param.requires_grad = False 
                    model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)
            if x[0]== 'densenet':    
                if self.model_type =='densenet121' :
                    model = models.densenet121(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = True 
                    classifier = nn.Sequential(
                                                nn.Linear(in_features=1024, out_features=256),
                                                nn.Dropout(p=0.3),  
                                                nn.ReLU(),
                                                nn.Linear(in_features=256, out_features=32),  
                                                nn.Dropout(p=0.3),
                                                nn.ReLU(),  
                                                nn.Linear(in_features=32, out_features=self.num_classes),
                                                nn.LogSoftmax(dim=1)  
                                                )

                    model.classifier = classifier   
                    #model.classifier = nn.Linear(model.classifier.in_features, self.num_classes)
                elif self.model_type =='densenet161' :
                    model = models.densenet161(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    classifier = nn.Sequential(
                                    nn.Linear(in_features=2208, out_features=1024),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.4),
                                    nn.Linear(in_features=1024, out_features=self.num_classes),
                                    nn.LogSoftmax(dim=1)  
                                    )
                                        
                    model.classifier = classifier     
                    #model.classifier = nn.Linear(model.classifier.in_features, self.num_classes)
                elif self.model_type == 'densenet169' :
                    from torchvision.models  import densenet169, DenseNet169_Weights
                    weights=DenseNet169_Weights.IMAGENET1K_V1
                    model = models.densenet169(weights=(weights,pretrained=self.pretrained)).to(device)    
                    #model = models.densenet169(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    classifier = nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(in_features=model.classifier.in_features, out_features=512, bias=True),
                                nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True), 
                                #nn.Dropout(p=0.5, inplace=False),
                                nn.Linear(in_features=512, out_features=256, bias=True),
                                nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True), 
                                nn.Dropout(p=0.25, inplace=False),
                                nn.Linear(in_features=256, out_features=self.num_classes, bias=True),
                                )     
                    model.classifier = classifier #nn.Linear(model.classifier.in_features, self.num_classes)
                elif self.model_type =='densenet201' :
                    from torchvision.models  import densenet201, DenseNet201_Weights
                    weights=DenseNet201_Weights.IMAGENET1K_V1
                    model = models.densenet201(weights=(weights,pretrained=self.pretrained)).to(device)    
                    for param in model.parameters():
                        param.requires_grad = True 
                    classifier = nn.Sequential(
                                    nn.Linear(in_features=model.classifier.in_features, out_features=1920, bias=True),
                                    nn.Linear(in_features=1920, out_features=1920, bias=True),
                                    nn.BatchNorm1d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=1920, out_features=1920, bias=True),
                                    nn.Linear(in_features=1920, out_features=1920, bias=True),
                                    nn.Linear(in_features=1920, out_features=1024, bias=True),
                                    nn.Linear(in_features=1024, out_features=1024, bias=True),
                                    nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=1024, out_features=512, bias=True),
                                    nn.Linear(in_features=512, out_features=512, bias=True),
                                    nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=512, out_features=256, bias=True),
                                    nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=256, out_features=128, bias=True),
                                    nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(in_features=128, out_features=self.num_classes, bias=True),
                                    )
                    model.classifier = classifier
            if x[0]=='googlenet':
                model = models.googlenet(pretrained=self.pretrained).to(device)
                for param in model.parameters():
                    param.requires_grad = False 
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            if x[0]== 'shufflenet':
                if self.model_type==  'shufflenet_v2_x0_5' :
                    model = models.shufflenet_v2_x0_5(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = False 
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
                elif self.model_type== 'shufflenet_v2_x1_0':
                    model = models.shufflenet_v2_x1_0(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = False 
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            if x[0]=='mobilenet' :   
                model = models.mobilenet_v2(pretrained=self.pretrained).to(device)
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
            if x[0]=='wide_resnet':   
                if 'wide_resnet50_2' == self.model_type:
                    model = models.wide_resnet50_2(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = False 
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
                elif self.model_type=='wide_resnet101_2':
                    model = models.wide_resnet101_2(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = False
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            if x[0]=='mnasnet':   
                if self.model_type == 'mnasnet0_5':
                    model = models.mnasnet0_5(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = False
                    model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
                elif self.model_type=='mnasnet1_0':
                    model = models.mnasnet1_0(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = False
                    model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes).double()           
            if x[0]=='efficientnet':   
                if self.model_type=='efficientnet-b0':
                    model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=self.num_classes).to(device)
                    for param in model.parameters():
                        param.requires_grad =True                     
                    model._fc = nn.Linear(model._fc.in_features,self.N_class).to(device)
                elif self.model_type=='efficientnet-b1':
                    model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=self.num_classes).to(device)
                    for param in model.parameters():
                        param.requires_grad =True                     
                    model._fc = nn.Linear(model._fc.in_features,self.N_class).to(device)   
                elif self.model_type=='efficientnet-b2':
                    model = EfficientNet.from_pretrained('efficientnet-b2',num_classes=self.num_classes).to(device)
                    for param in model.parameters():
                        param.requires_grad =True   
                    model._fc = nn.Linear(model._fc.in_features,self.N_class).to(device)                                  
                elif self.model_type=='efficientnet-b3':
                    model = EfficientNet.from_pretrained('efficientnet-b3',num_classes=self.num_classes).to(device)
                    for param in model.parameters():
                        param.requires_grad =True                     
                    model._fc = nn.Linear(model._fc.in_features,self.N_class).to(device)        
                elif self.model_type=='efficientnet-b4':
                    model = EfficientNet.from_pretrained('efficientnet-b4',num_classes=self.num_classes).to(device)
                    for param in model.parameters():
                        param.requires_grad =True                     
                    model._fc = nn.Linear(model._fc.in_features,self.N_class).to(device)
                elif self.model_type=='efficientnet-b5':
                    model = EfficientNet.from_pretrained('efficientnet-b5',num_classes=self.num_classes).to(device)
                    for param in model.parameters():
                        param.requires_grad =True                     
                    model._fc = nn.Linear(model._fc.in_features,self.N_class).to(device)                   
                elif self.model_type=='efficientnet-b6':
                    model = EfficientNet.from_pretrained('efficientnet-b6',num_classes=self.num_classes).to(device)
                    for param in model.parameters():
                        param.requires_grad =True                     
                    model._fc = nn.Linear(model._fc.in_features,self.N_class).to(device)
                elif self.model_type=='efficientnet-b7':
                    model = EfficientNet.from_pretrained('efficientnet-b7',num_classes=self.num_classes).to(device)
                    for param in model.parameters():
                        param.requires_grad =True                     
                    model._fc = nn.Linear(model._fc.in_features,self.N_class).to(device)    
        elif int(self.ImageShape)==self.commonShapes[1]:
            if x=='squeezenet':
                if self.model_type =='squeezenet1_0':
                    model = models.squeezenet1_0(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = False 
                    model.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1,1), stride=(1,1))
                    model.num_classes = self.num_classes
            elif self.model_type=='squeezenet1_1':
                model = models.squeezenet1_1(pretrained=self.pretrained).to(device)
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1,1), stride=(1,1))
                model.num_classes = self.num_classes
        elif int(self.ImageShape)==self.commonShapes[2]:
            if x=='resnext':
                if self.model_type=='resnext50_32x4d' :
                    model = models.resnext50_32x4d(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = False 
                    model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
                elif self.model_type=='resnext101_32x8d' :
                    model = models.resnext101_32x8d(pretrained=self.pretrained).to(device).double()
                    for param in model.parameters():
                        param.requires_grad = False 
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)        
        elif int(self.ImageShape)==self.commonShapes[3]:
            if x=='inception' :
                model = models.inception_v3(pretrained=self.pretrained).to(device)
                for param in model.parameters():
                    param.requires_grad = False
                model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, self.num_classes)
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)          
        else:
            raise AssertionError(f'ImageShape "{self.ImageShape}" is not a valid size for an image !,plase insert a Valid from : {commonShapes} more info check https://medium.com/analytics-vidhya/how-to-pick-the-optimal-image-size-for-training-convolution-neural-network-65702b880f05')
        return model
    
    def optimizer(self,model):
        criterion = nn.CrossEntropyLoss() 
        if int(self.ImageShape) in self.commonShapes[2:] :
            #optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        elif int(self.ImageShape)==self.commonShapes[1]:
            optimizer=torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
            # Decay LR by a factor of 0.1 every 7 epochs
            exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        elif int(self.ImageShape)==self.commonShapes[0]:
            optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
            exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience =  5, mode = 'max', verbose=True)       
            criterion = nn.NLLLoss()
        return   criterion,optimizer,exp_lr_scheduler

#np.random.seed(37)
#torch.manual_seed(37)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

#num_classes = 3
#pretrained = True
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#EpochProgress = namedtuple('EpochProgress', 'epoch, loss, accuracy')

#transform = transforms.Compose([Resize(224), ToTensor()])
#image_folder = datasets.ImageFolder('./shapes/train', transform=transform)
#dataloader = DataLoader(image_folder, batch_size=4, shuffle=True, num_workers=4)
    

