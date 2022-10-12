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
    commonShapes=[224,240,288,299,300,380,384,456,512,518,528,600]
    
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
        
        models_list=['resnet','regnet','alexnet','vgg','vision_transformer','densenet','googlenet','shufflenet',
                     'mobilenet_v2','mobilenet_v3','wide_resnet','efficientnet','efficientnet_v2','ConvNeXt','swin_transformer','squeezenet',
                     'mnasnet','resnext','inception']
        x=[i for i in models_list if i in self.model_type]
        model = None
        if int(self.ImageShape)==self.commonShapes[0]:
            if x[0]== 'resnet':
                if self.model_type =='resnet18':
                    from torchvision.models  import resnet18, ResNet18_Weights
                    weights=ResNet18_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.resnet18(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    model.fc = nn.Linear(model.fc.in_features,self.num_classes )
                elif  self.model_type=='resnet34'  :
                    from torchvision.models  import resnet34, ResNet34_Weights
                    weights=ResNet34_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.resnet34(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
                elif self.model_type== 'resnet50' :
                    from torchvision.models  import resnet50, ResNet50_Weights
                    weights=ResNet50_Weights.IMAGENET1K_V2
                    pretrained=self.pretrained
                    model = models.resnet50(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    classifier = nn.Sequential(
                                    nn.Linear(in_features=model.fc.in_features, out_features=1024),
                                    nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  
                                    nn.ReLU(),
                                    nn.Linear(in_features=1024, out_features=256),
                                    nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  
                                    nn.ReLU(),
                                    nn.Linear(in_features=256, out_features=self.num_classes),
                                    nn.LogSoftmax(dim=1)  
                                    )    
                    model.fc = classifier     
                elif self.model_type=='resnet101'  :
                    from torchvision.models  import resnet101, ResNet101_Weights
                    weights=ResNet101_Weights.IMAGENET1K_V2
                    pretrained=self.pretrained
                    model = models.resnet101(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
                elif self.model_type== 'resnet152' :
                    from torchvision.models  import resnet152, ResNet152_Weights
                    weights=ResNet152_Weights.IMAGENET1K_V2
                    pretrained=self.pretrained
                    model = models.resnet152(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    classifier = nn.Sequential(
                                    nn.Flatten(),
                                    nn.Linear(in_features=model.fc.in_features, out_features=1024, bias=True),
                                    nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(in_features=1024, out_features=512, bias=True),
                                    nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(in_features=512, out_features=256, bias=True),
                                    nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=256, out_features=self.num_classes, bias=True),
                                    nn.LogSoftmax(dim=1) ,
                                    )
                    model.fc = classifier  
            elif x[0]== 'regnet':
                if self.model_type =='regnet_x_16gf':
                    from torchvision.models  import regnet_x_16gf, RegNet_X_16GF_Weights
                    weights=RegNet_X_16GF_Weights.IMAGENET1K_V2
                    pretrained=self.pretrained
                    model = models.regnet_x_16gf(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                            param.requires_grad = True
                    classifier = nn.Sequential(
                                    nn.Linear(in_features=model.fc.in_features, out_features=1024, bias=True),
                                    nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=1024, out_features=512, bias=True),
                                    nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=512, out_features=256, bias=True),
                                    nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=256, out_features=self.num_classes, bias=True),
                                    nn.LogSoftmax(dim=1) ,
                                    )
                    model.fc = classifier                        
                elif  self.model_type=='regnet_x_1_6gf':
                    from torchvision.models  import regnet_x_1_6gf, RegNet_X_1_6GF_Weights
                    weights=RegNet_X_1_6GF_Weights.IMAGENET1K_V2
                    pretrained=self.pretrained
                    model = models.regnet_x_1_6gf(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
                elif self.model_type== 'regnet_x_32gf' :
                    from torchvision.models  import regnet_x_32gf, RegNet_X_32GF_Weights
                    weights=RegNet_X_32GF_Weights.IMAGENET1K_V2
                    pretrained=self.pretrained
                    model = models.regnet_x_32gf(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
                elif self.model_type=='regnet_x_3_2gf':
                    from torchvision.models  import regnet_x_3_2gf, RegNet_X_3_2GF_Weights
                    weights=RegNet_X_3_2GF_Weights.IMAGENET1K_V2
                    pretrained=self.pretrained
                    model = models.regnet_x_3_2gf(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
                elif self.model_type== 'regnet_x_400mf' :
                    from torchvision.models  import regnet_x_400mf, RegNet_X_400MF_Weights
                    weights=RegNet_X_400MF_Weights.IMAGENET1K_V2
                    pretrained=self.pretrained
                    model = models.regnet_x_400mf(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)   
                elif self.model_type== 'regnet_x_800mf' :
                    from torchvision.models  import regnet_x_800mf, RegNet_X_800MF_Weights
                    weights=RegNet_X_800MF_Weights.IMAGENET1K_V2
                    pretrained=self.pretrained
                    model = models.regnet_x_800mf(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
                elif self.model_type=='regnet_x_8gf':
                    from torchvision.models  import regnet_x_8gf, RegNet_X_8GF_Weights
                    weights=RegNet_X_8GF_Weights.IMAGENET1K_V2
                    pretrained=self.pretrained
                    model = models.regnet_x_8gf(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
                elif self.model_type== 'regnet_y_128gf' :
                    from torchvision.models  import regnet_y_128gf, RegNet_Y_128GF_Weights
                    weights=RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_LINEAR_V1
                    pretrained=self.pretrained
                    model = models.regnet_y_128gf(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
                elif self.model_type=='regnet_y_16gf':
                    from torchvision.models  import regnet_y_16gf, RegNet_Y_16GF_Weights
                    weights=RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_LINEAR_V1
                    pretrained=self.pretrained
                    model = models.regnet_y_16gf(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes) 
                elif self.model_type== 'regnet_y_1_6gf' :
                    from torchvision.models  import regnet_y_1_6gf, RegNet_Y_1_6GF_Weights.IMAGENET1K_V2.
                    weights=RegNet_Y_1_6GF_Weights.IMAGENET1K_V2
                    pretrained=self.pretrained
                    model = models.regnet_y_1_6gf(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes) 
                elif self.model_type== 'regnet_y_32gf' :
                    from torchvision.models  import regnet_y_32gf, RegNet_Y_32GF_Weights
                    weights=RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_LINEAR_V1
                    pretrained=self.pretrained
                    model = models.regnet_y_32gf(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes) 
                elif self.model_type== 'regnet_y_32gf' :
                    from torchvision.models  import regnet_y_32gf, RegNet_Y_32GF_Weights
                    weights=RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1
                    pretrained=self.pretrained
                    model = models.regnet_y_32gf(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
                elif self.model_type== 'regnet_y_3_2gf' :
                    from torchvision.models  import regnet_y_3_2gf, RegNet_Y_3_2GF_Weights
                    weights=RegNet_Y_3_2GF_Weights.IMAGENET1K_V2
                    pretrained=self.pretrained
                    model = models.regnet_y_3_2gf(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes) 
                elif self.model_type== 'regnet_y_400mf' :
                    from torchvision.models  import regnet_y_400mf, RegNet_Y_400MF_Weights
                    weights=RegNet_Y_400MF_Weights.IMAGENET1K_V2
                    pretrained=self.pretrained
                    model = models.regnet_y_400mf(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes) 
                elif self.model_type== 'regnet_y_800mf' :
                    from torchvision.models  import regnet_y_800mf,RegNet_Y_800MF_Weights
                    weights=RegNet_Y_800MF_Weights.IMAGENET1K_V2
                    pretrained=self.pretrained
                    model = models.regnet_y_800mf(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)  
                elif self.model_type== 'regnet_y_8gf' :
                    from torchvision.models  import regnet_y_8gf,RegNet_Y_8GF_Weights
                    weights=RegNet_Y_8GF_Weights.IMAGENET1K_V2
                    pretrained=self.pretrained
                    model = models.regnet_y_8gf(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)                                                                                           
            elif x[0]=='alexnet':
                from torchvision.models  import alexnet, AlexNet_Weights
                weights=AlexNet_Weights.IMAGENET1K_V1
                pretrained=self.pretrained
                model = models.alexnet(weights=(weights,pretrained)).to(device) 
                for param in model.parameters():
                    param.requires_grad = True
                classifier =nn.Sequential(   
                            nn.Linear(in_features=model.classifier[6].in_features, out_features=4096),
                            nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.ReLU(),
                            nn.Linear(in_features=4096, out_features=2048),
                            nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.ReLU(),   
                            nn.Linear(in_features=2048, out_features=1024),
                            nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  
                            nn.ReLU(),
                            nn.Linear(in_features=1024, out_features=512),
                            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  
                            nn.ReLU(),  
                            nn.Linear(in_features=512, out_features=256),  
                            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  
                            nn.ReLU(),
                            nn.Linear(in_features=256, out_features=128),  
                            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  
                            nn.ReLU(), 
                            nn.Linear(in_features=128, out_features=self.num_classes), 
                            )    
                model.classifier[6]=classifier  
            elif x[0]== 'vgg' :
                if self.model_type=='vgg11' :
                    from torchvision.models  import vgg11,VGG11_Weights
                    weights=VGG11_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.vgg11(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = False
                    classifier =nn.Sequential(
                                nn.Flatten(),
                                nn.BatchNorm1d(25088, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=model.classifier.in_features, out_features=16384, bias=True),
                                nn.Dropout(p=0.65, inplace=False),
                                nn.ReLU(inplace=True),
                                nn.BatchNorm1d(16384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=16384, out_features=8192, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.55, inplace=False),
                                nn.BatchNorm1d(8192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=8192, out_features=8192, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.45, inplace=False),
                                nn.BatchNorm1d(8192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=8192, out_features=4096, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.45, inplace=False),
                                nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=4096, out_features=4096, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.30, inplace=False),
                                nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=4096, out_features=2048, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.20, inplace=False),
                                nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=2048, out_features=1024, bias=True),
                                nn.ReLU(inplace=True), 
                                nn.Dropout(p=0.20, inplace=False),
                                nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=1024, out_features=512, bias=True),
                                nn.ReLU(inplace=True), 
                                nn.Dropout(p=0.15, inplace=False),
                                nn.Linear(in_features=512, out_features=self.num_classes, bias=True),
                                nn.LogSoftmax(dim=1),
                                )
                    model.classifier = classifier    
                elif self.model_type =='vgg11_bn' :
                    from torchvision.models  import vgg11_bn, VGG11_BN_Weights
                    weights=VGG11_BN_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.vgg11_bn(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    classifier =nn.Sequential(
                                nn.Linear(in_features=model.classifier.in_features, out_features=16384, bias=True),
                                nn.BatchNorm1d(16384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=16384, out_features=4096, bias=True),
                                nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=4096, out_features=2048, bias=True),
                                nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=2048, out_features=512, bias=True),
                                nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=512, out_features=256, bias=True),
                                nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True), 
                                nn.Linear(in_features=256, out_features=self.num_classes, bias=True),
                                nn.LogSoftmax(dim=1) ,
                                )
                    model.classifier = classifier
                elif self.model_type== 'vgg13' :
                    from torchvision.models  import vgg13, VGG13_Weights
                    weights=VGG13_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.vgg13(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True    
                    classifier =nn.Sequential(
                                nn.Flatten(),
                                nn.BatchNorm1d(25088, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=model.classifier.in_features, out_features=16384, bias=True),
                                nn.Dropout(p=0.65, inplace=False),
                                nn.ReLU(inplace=True),
                                nn.BatchNorm1d(16384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=16384, out_features=8192, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.55, inplace=False),
                                nn.BatchNorm1d(8192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=8192, out_features=8192, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.45, inplace=False),
                                nn.BatchNorm1d(8192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=8192, out_features=4096, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.45, inplace=False),
                                nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=4096, out_features=4096, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.30, inplace=False),
                                nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=4096, out_features=2048, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.20, inplace=False),
                                nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=2048, out_features=1024, bias=True),
                                nn.ReLU(inplace=True), 
                                nn.Dropout(p=0.20, inplace=False),
                                nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=1024, out_features=512, bias=True),
                                nn.ReLU(inplace=True), 
                                nn.Dropout(p=0.15, inplace=False),
                                nn.Linear(in_features=512, out_features=self.num_classes, bias=True),
                                nn.LogSoftmax(dim=1),
                                )
                    model.classifier = classifier    
                elif self.model_type == 'vgg13_bn' :
                    from torchvision.models  import vgg13_bn, VGG13_BN_Weights
                    weights=VGG13_BN_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.vgg13_bn(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True 
                    classifier =nn.Sequential(
                                nn.Flatten(),
                                nn.BatchNorm1d(25088, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=model.classifier.in_features, out_features=16384, bias=True),
                                nn.Dropout(p=0.65, inplace=False),
                                nn.ReLU(inplace=True),
                                nn.BatchNorm1d(16384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=16384, out_features=8192, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.55, inplace=False),
                                nn.BatchNorm1d(8192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=8192, out_features=8192, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.45, inplace=False),
                                nn.BatchNorm1d(8192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=8192, out_features=4096, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.45, inplace=False),
                                nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=4096, out_features=4096, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.30, inplace=False),
                                nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=4096, out_features=2048, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.20, inplace=False),
                                nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=2048, out_features=1024, bias=True),
                                nn.ReLU(inplace=True), 
                                nn.Dropout(p=0.20, inplace=False),
                                nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=1024, out_features=512, bias=True),
                                nn.ReLU(inplace=True), 
                                nn.Dropout(p=0.15, inplace=False),
                                nn.Linear(in_features=512, out_features=self.num_classes, bias=True),
                                nn.LogSoftmax(dim=1),
                                )    
                    model.classifier = classifier
                elif self.model_type == 'vgg16' :
                    from torchvision.models  import vgg16, VGG16_Weights
                    weights=VGG16_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.vgg16(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True 
                    classifier =nn.Sequential(
                                nn.Flatten(),
                                nn.BatchNorm1d(25088, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=model.classifier.in_features, out_features=16384, bias=True),
                                nn.Dropout(p=0.65, inplace=False),
                                nn.ReLU(inplace=True),
                                nn.BatchNorm1d(16384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=16384, out_features=8192, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.55, inplace=False),
                                nn.BatchNorm1d(8192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=8192, out_features=8192, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.45, inplace=False),
                                nn.BatchNorm1d(8192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=8192, out_features=4096, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.45, inplace=False),
                                nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=4096, out_features=4096, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.30, inplace=False),
                                nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=4096, out_features=2048, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.20, inplace=False),
                                nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=2048, out_features=1024, bias=True),
                                nn.ReLU(inplace=True), 
                                nn.Dropout(p=0.20, inplace=False),
                                nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=1024, out_features=512, bias=True),
                                nn.ReLU(inplace=True), 
                                nn.Dropout(p=0.15, inplace=False),
                                nn.Linear(in_features=512, out_features=self.num_classes, bias=True),
                                nn.LogSoftmax(dim=1),
                                )    
                    model.classifier = classifier
                elif self.model_type== 'vgg16_bn' :
                    from torchvision.models  import vgg16_bn, VGG16_BN_Weights
                    weights=VGG16_BN_Weights.IMAGENET1K_V1
                    model = models.vgg16_bn(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = True 
                    classifier =nn.Sequential(
                                nn.Flatten(),
                                nn.BatchNorm1d(25088, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=model.classifier.in_features, out_features=16384, bias=True),
                                nn.Dropout(p=0.65, inplace=False),
                                nn.ReLU(inplace=True),
                                nn.BatchNorm1d(16384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=16384, out_features=8192, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.55, inplace=False),
                                nn.BatchNorm1d(8192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=8192, out_features=8192, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.45, inplace=False),
                                nn.BatchNorm1d(8192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=8192, out_features=4096, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.45, inplace=False),
                                nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=4096, out_features=4096, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.30, inplace=False),
                                nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=4096, out_features=2048, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.20, inplace=False),
                                nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=2048, out_features=1024, bias=True),
                                nn.ReLU(inplace=True), 
                                nn.Dropout(p=0.20, inplace=False),
                                nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=1024, out_features=512, bias=True),
                                nn.ReLU(inplace=True), 
                                nn.Dropout(p=0.15, inplace=False),
                                nn.Linear(in_features=512, out_features=self.num_classes, bias=True),
                                nn.LogSoftmax(dim=1),
                                )    
                    model.classifier = classifier                  
                elif self.model_type=='vgg19' :
                    from torchvision.models  import vgg19, VGG19_Weights
                    weights=VGG19_Weights.IMAGENET1K_V1
                    model = models.vgg19(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = False 
                    classifier =nn.Sequential(
                                nn.Flatten(),
                                nn.BatchNorm1d(25088, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=model.classifier.in_features, out_features=16384, bias=True),
                                nn.Dropout(p=0.65, inplace=False),
                                nn.ReLU(inplace=True),
                                nn.BatchNorm1d(16384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=16384, out_features=8192, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.55, inplace=False),
                                nn.BatchNorm1d(8192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=8192, out_features=8192, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.45, inplace=False),
                                nn.BatchNorm1d(8192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=8192, out_features=4096, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.45, inplace=False),
                                nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=4096, out_features=4096, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.30, inplace=False),
                                nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=4096, out_features=2048, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.20, inplace=False),
                                nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=2048, out_features=1024, bias=True),
                                nn.ReLU(inplace=True), 
                                nn.Dropout(p=0.20, inplace=False),
                                nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=1024, out_features=512, bias=True),
                                nn.ReLU(inplace=True), 
                                nn.Dropout(p=0.15, inplace=False),
                                nn.Linear(in_features=512, out_features=self.num_classes, bias=True),
                                nn.LogSoftmax(dim=1),
                                )    
                    model.classifier = classifier 
                elif self.model_type=='vgg19_bn':
                    from torchvision.models  import vgg19, VGG19_BN_Weights
                    weights=VGG19_BN_Weights.IMAGENET1K_V1
                    model = models.vgg19bn(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = False 
                    classifier =nn.Sequential(
                                nn.Flatten(),
                                nn.BatchNorm1d(25088, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=model.classifier.in_features, out_features=16384, bias=True),
                                nn.Dropout(p=0.65, inplace=False),
                                nn.ReLU(inplace=True),
                                nn.BatchNorm1d(16384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=16384, out_features=8192, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.55, inplace=False),
                                nn.BatchNorm1d(8192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=8192, out_features=8192, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.45, inplace=False),
                                nn.BatchNorm1d(8192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=8192, out_features=4096, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.45, inplace=False),
                                nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=4096, out_features=4096, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.30, inplace=False),
                                nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=4096, out_features=2048, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.20, inplace=False),
                                nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=2048, out_features=1024, bias=True),
                                nn.ReLU(inplace=True), 
                                nn.Dropout(p=0.20, inplace=False),
                                nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.Linear(in_features=1024, out_features=512, bias=True),
                                nn.ReLU(inplace=True), 
                                nn.Dropout(p=0.15, inplace=False),
                                nn.Linear(in_features=512, out_features=self.num_classes, bias=True),
                                nn.LogSoftmax(dim=1),
                                )
                    model.classifier = classifier 
            elif x[0]== 'vision_transformer':    
                if self.model_type =='vit_b_16' :
                    from torchvision.models  import vit_b_16, ViT_B_16_Weights
                    weights=ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
                    pretrained=self.pretrained
                    model = models.vit_b_16(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = True 
                    classifier= nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(in_features=model.fc.in_features, out_features=512, bias=True),
                                nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True), 
                                nn.Linear(in_features=512, out_features=256, bias=True),
                                nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True), 
                                nn.Linear(in_features=256, out_features=self.num_classes, bias=True),
                                )
                    model.fc=classifier  
                elif self.model_type =='vit_b_32' :
                    from torchvision.models  import vit_b_32, ViT_B_32_Weights
                    weights=ViT_B_32_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.vit_b_32(weights=(weights,pretrained)).to(device) 
                    for param in model.parameters():
                        param.requires_grad = True
                    classifier = nn.Sequential(
                                    nn.Linear(in_features=model.classifier.in_features, out_features=1024),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.4),
                                    nn.Linear(in_features=1024, out_features=self.num_classes),
                                    nn.LogSoftmax(dim=1)  
                                    )                     
                    model.fc = classifier     
                    #model.classifier = nn.Linear(model.classifier.in_features, self.num_classes)
                elif self.model_type == 'vit_h_14' :
                    from torchvision.models  import vit_h_14, ViT_H_14_Weights
                    weights=ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1
                    pretrained=self.pretrained
                    model = models.vit_h_14(weights=(weights,pretrained)).to(device)    
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
                elif self.model_type =='vit_l_16' :
                    from torchvision.models  import vit_l_16, ViT_L_16_Weights
                    weights=ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
                    pretrained=self.pretrained
                    model = models.vit_l_16(weights=(weights,pretrained)).to(device)    
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
                elif self.model_type =='vit_l_32' :
                    from torchvision.models  import vit_l_32, ViT_L_32_Weights
                    weights=ViT_L_32_Weights
                    pretrained=self.pretrained
                    model = models.vit_l_32(weights=(weights,pretrained)).to(device)    
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
            elif x[0]== 'densenet':    
                if self.model_type =='densenet121' :
                    from torchvision.models  import densenet121, DenseNet121_Weights
                    weights=DenseNet121_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.densenet121(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = True 
                    classifier = nn.Sequential(
                                                nn.Linear(in_features=model.classifier.in_features, out_features=256),
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
                    from torchvision.models  import densenet161, DenseNet161_Weights
                    weights=DenseNet161_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.densenet169(weights=(weights,pretrained)).to(device) 
                    for param in model.parameters():
                        param.requires_grad = True
                    classifier = nn.Sequential(
                                    nn.Linear(in_features=model.classifier.in_features, out_features=1024),
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
                    pretrained=self.pretrained
                    model = models.densenet169(weights=(weights,pretrained)).to(device)    
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
                    pretrained=self.pretrained
                    model = models.densenet201(weights=(weights,pretrained)).to(device)    
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
            elif x[0]=='googlenet':
                from torchvision.models import googlenet,GoogLeNet_Weights
                weights=GoogLeNet_Weights.IMAGENET1K_V1
                pretrained=self.pretrained
                model = models.googlenet(weights=(weights,pretrained)).to(device) 
                for param in model.parameters():
                    param.requires_grad = False 
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            elif x[0]== 'shufflenet':
                if self.model_type==  'shufflenet_v2_x0_5' :
                    from torchvision.models import shufflenet_v2_x0_5,ShuffleNet_V2_X0_5_Weights
                    weights=ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.shufflenet_v2_x0_5(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True 
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
                elif self.model_type== 'shufflenet_v2_x1_5':
                    from torchvision.models import shufflenet_v2_x1_5,ShuffleNet_V2_X1_0_Weights
                    weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.shufflenet_v2_x1_5(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
                elif self.model_type==  'shufflenet_v2_x2_0' :
                    from torchvision.models import shufflenet_v2_x0_5,ShuffleNet_V2_X2_0_Weights
                    weights=ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.shufflenet_v2_x2_0(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True 
                    model.fc = nn.Linear(model.fc.in_features, self.num_class)
                elif self.model_type==  'shufflenet_v2_x2_0' :
                    from torchvision.models import shufflenet_v2_x0_5,GoogLeNet_Weights
                    weights=GoogLeNet_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.shufflenet_v2_x0_5(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True 
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)        
            elif x[0]=='mobilenet_v2' :   
                from torchvision.models import mobilenet_v2,MobileNet_V2_Weights
                weights=MobileNet_V2_Weights.IMAGENET1K_V2
                pretrained=self.pretrained
                model = models.mobilenet_v2(weights=(weights,pretrained)).to(device)
                for param in model.parameters():
                    param.requires_grad = True 
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
            elif x[0]=='mobilenet_v3':   
                if self.model_type == 'mobilenet_v3_small':
                    from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
                    weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.mobilenet_v3_small(weights=(weights,pretrained)).to(device) 
                    for param in model.parameters():
                        param.requires_grad = True
                    model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
                elif self.model_type == 'mobilenet_v3_large':
                    from torchvision.models import mobilenet_v3_large,MobileNet_V3_Large_Weights
                    weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2
                    pretrained=self.pretrained
                    model = models.mobilenet_v3_large(weights=(weights,pretrained)).to(device) 
                    for param in model.parameters():
                        param.requires_grad = True
                    model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
            elif x[0]=='wide_resnet':   
                if self.model_type=='wide_resnet50_2' :
                    from torchvision.models import wide_resnet50_2,Wide_ResNet50_2_Weights
                    weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2
                    pretrained=self.pretrained
                    model = models.wide_resnet50_2(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True 
                    classifier =nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(in_features=model.fc.in_features, out_features=1024, bias=True),
                                nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=1024, out_features=512, bias=True),
                                nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=512, out_features=256, bias=True),
                                nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True), 
                                nn.Linear(in_features=256, out_features=self.num_classes, bias=True),
                                nn.LogSoftmax(dim=1) ,
                                )
                    model.fc = classifier    
                elif self.model_type=='wide_resnet101_2':
                    from torchvision.models import wide_resnet101_2,Wide_ResNet101_2_Weights
                    weights=Wide_ResNet101_2_Weights.IMAGENET1K_V2
                    pretrained=self.pretrained
                    model = models.wide_resnet101_2(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    classifier =nn.Sequential(
                                nn.Linear(in_features=model.fc.in_features, out_features=2048),
                                nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(),
                                nn.Linear(in_features=2048, out_features=1024),
                                nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(),
                                nn.Linear(in_features=1024, out_features=1024),
                                nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(),
                                nn.Linear(in_features=1024, out_features=512),
                                nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(), 
                                nn.Linear(in_features=512, out_features=256),  
                                nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  
                                nn.ReLU(),
                                nn.Linear(in_features=256, out_features=128),
                                nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),  
                                nn.ReLU(),
                                nn.Linear(in_features=128, out_features=self.num_classes), 
                                )    
                    model.fc=classifier    
            elif x[0]=='mnasnet':   
                if self.model_type == 'mnasnet0_5':
                    from torchvision.models import mnasnet0_5,MNASNet0_5_Weights
                    weights=MNASNet0_5_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.mnasnet0_5(weights=(weights,pretrained)).to(device) 
                    for param in model.parameters():
                        param.requires_grad = True
                    model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
                elif self.model_type == 'mnasnet0_75':
                    from torchvision.models import mnasnet0_75,MNASNet0_75_Weights
                    weights=MNASNet0_75_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.mnasnet0_75(weights=(weights,pretrained)).to(device) 
                    for param in model.parameters():
                        param.requires_grad = True
                    model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
                elif self.model_type=='mnasnet1_0':
                    from torchvision.models import mnasnet1_0,MNASNet1_0_Weights
                    weights=MNASNet1_0_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.mnasnet1_0(weights=(weights,pretrained)).to(device) 
                    for param in model.parameters():
                        param.requires_grad = True
                    model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes).double()           
                elif self.model_type == 'mnasnet1_3':
                    from torchvision.models import mnasnet1_3,MNASNet1_3_Weights
                    weights=MNASNet1_3_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.mnasnet1_3(weights=(weights,pretrained)).to(device) 
                    for param in model.parameters():
                        param.requires_grad = True
                    model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)    
            elif x[0]=='efficientnet':   
                if self.model_type=='efficientnet-b0':
                    from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights
                    weights=EfficientNet_B0_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.efficientnet_b0(weights=(weights,pretrained)).to(device) 
                    for param in model.parameters():
                        param.requires_grad =True     
                    classifier = nn.Sequential(
                                    nn.Flatten(),
                                    nn.Linear(in_features=model.classifier.in_features, out_features=512, bias=True),
                                    nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=512, out_features=256, bias=True),
                                    nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=256, out_features=self.N_class, bias=True),
                                    nn.LogSoftmax(dim=1) ,
                                    ) 
                    model.classifier = classifier                                                                                          
            elif x[0]=='efficientnet_v2':  
                if self.model_type=='efficientnet_v2_s':
                    from torchvision.models import efficientnet_v2_s,EfficientNet_V2_S_Weights
                    weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.efficientnet_v2_s(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad =True  
                    classifier =nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(in_features=model.classifier.in_features, out_features=512, bias=True),
                                nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True), 
                                nn.Linear(in_features=512, out_features=256, bias=True),
                                nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True), 
                                nn.Linear(in_features=256, out_features=self.N_class, bias=True),
                                nn.LogSoftmax(dim=1) ,
                                )
                    model.classifier = classifier 
                elif self.model_type=='efficientnet_v2_m':
                    from torchvision.models import efficientnet_v2_m,EfficientNet_V2_M_Weights
                    weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.efficientnet_v2_m(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad =True  
                    classifier =nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(in_features=model.classifier.in_features, out_features=512, bias=True),
                                nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True), 
                                nn.Linear(in_features=512, out_features=256, bias=True),
                                nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True), 
                                nn.Linear(in_features=256, out_features=self.N_class, bias=True),
                                nn.LogSoftmax(dim=1) ,
                               )
                    model.classifier = classifier
                elif self.model_type=='efficientnet_v2_l':
                    from torchvision.models import efficientnet_v2_l,EfficientNet_V2_L_Weights
                    weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.efficientnet_v2_l(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad =True  
                    classifier= nn.Sequential(#not correct
                                nn.Flatten(),
                                nn.Linear(in_features=model.classifier.in_features, out_features=512, bias=True),
                                nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.ReLU(inplace=True), 
                                        nn.Linear(in_features=512, out_features=256, bias=True),
                                        nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.ReLU(inplace=True), 
                                        nn.Linear(in_features=256, out_features=self.N_class, bias=True),
                                        nn.LogSoftmax(dim=1) ,
                                        )
                    model.classifier = classifier                  
            elif x[0]=='convnext':  
                if self.model_type=='convnext_small':
                    from torchvision.models import coconvnext_small,ConvNeXt_Small_Weights
                    weights=ConvNeXt_Small_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.convnext_small(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad =True  
                    classifier=nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(in_features=768, out_features=512, bias=True),
                                nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True), 
                                nn.Linear(in_features=512, out_features=256, bias=True),
                                nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True), 
                                nn.Linear(in_features=256, out_features=2, bias=True),         
                                )
                    model.classifier = classifier
                elif self.model_type=='convnext_base':
                    from torchvision.models import convnext_base,ConvNeXt_Base_Weights
                    weights=ConvNeXt_Base_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.convnext_base(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad =True  
                    classifier= nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(in_features=1024, out_features=512, bias=True),
                                nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True), 
                                nn.Linear(in_features=512, out_features=256, bias=True),
                                nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True), 
                                nn.Linear(in_features=256, out_features=2, bias=True),                                     
                                )
                    model.classifier = classifier 
                elif self.model_type=='convnext_large':
                    from torchvision.models import convnext_large,ConvNeXt_Large_Weights
                    weights=ConvNeXt_Large_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.convnext_large(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad =True  
                    classifier = nn.Sequential(#not correct
                                    nn.Flatten(),
                                    nn.Linear(in_features=model.classifier.in_features, out_features=512, bias=True),
                                    nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=512, out_features=256, bias=True),
                                    nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=256, out_features=self.N_class, bias=True),
                                    nn.LogSoftmax(dim=1) ,
                                    )
                    model.classifier = classifier 
            elif x[0]=='swin_transformer':  
                if self.model_type=='swin_t':
                    from torchvision.models import swin_t,Swin_T_Weights
                    weights=Swin_T_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.swin_t(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad =True  
                    classifier =nn.Sequential(#not correct
                                        nn.Flatten(),
                                        nn.Linear(in_features=model.classifier.in_features, out_features=512, bias=True),
                                        nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.ReLU(inplace=True), 
                                        nn.Linear(in_features=512, out_features=256, bias=True),
                                        nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.ReLU(inplace=True), 
                                        nn.Linear(in_features=256, out_features=self.N_class, bias=True),                                       
                                        )
                elif self.model_type=='swin_s':
                    from torchvision.models import swin_s,Swin_S_Weights
                    weights=Swin_S_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.swin_b(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad =True  
                    n_inputs = model.head.in_features
                    model.head =nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(in_features=n_inputs, out_features=512, bias=True),
                                nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True), 
                                nn.Linear(in_features=512, out_features=256, bias=True),
                                nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True), 
                                nn.Linear(in_features=256, out_features=self.N_class, bias=True),
                                )     
                elif self.model_type=='swin_b':
                    from torchvision.models import swin_b,Swin_B_Weights
                    weights=Swin_B_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.swin_b(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad =True  
                    n_inputs = model.head.in_features
                    model.head =nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(in_features=n_inputs, out_features=512, bias=True),
                                nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True), 
                                nn.Linear(in_features=512, out_features=256, bias=True),
                                nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True), 
                                nn.Linear(in_features=256, out_features=self.N_class, bias=True),
                                )                                              
            elif x[0]=='squeezenet':
                if self.model_type =='squeezenet1_0':
                    from torchvision.models import squeezenet1_0,SqueezeNet1_0_Weights
                    weights=SqueezeNet1_0_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.squeezenet1_0(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True 
                    model.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1,1), stride=(1,1))
                    model.num_classes = self.num_classes
                elif self.model_type=='squeezenet1_1':
                    from torchvision.models import squeezenet1_1,SqueezeNet1_1_Weights
                    weights=SqueezeNet1_1_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.squeezenet1_1(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True 
                    model.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1,1), stride=(1,1))
                    model.num_classes = self.num_classes 
            elif x[0]=='resnext':
                if self.model_type=='resnext50_32x4d' :
                    from torchvision.models import resnext50_32x4d,ResNeXt50_32X4D_Weights
                    weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.resnext50_32x4d(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True 
                    classifier = nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(in_features=model.classifier[1].in_features, out_features=1024, bias=True),
                                nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=1024, out_features=512, bias=True),
                                nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=512, out_features=256, bias=True),
                                nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True), 
                                nn.Linear(in_features=256, out_features=self.num_classes, bias=True),
                                nn.LogSoftmax(dim=1) ,
                                )
                    model.fc = classifier
                elif self.model_type=='resnext101_32x8d' :
                    from torchvision.models import resnext101_32x8d,ResNeXt101_32X8D_Weights
                    weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2
                    pretrained=self.pretrained
                    model = models.resnext101_32x8d(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True 
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)                 
        elif int(self.ImageShape)==self.commonShapes[1]:
            if x[0]=='efficientnet': 
                if self.model_type=='efficientnet-b1':
                        from torchvision.models import efficientnet_b1,EfficientNet_B1_Weights
                        weights=EfficientNet_B1_Weights.IMAGENET1K_V1
                        pretrained=self.pretrained
                        model = models.efficientnet_b1(weights=(weights,pretrained)).to(device) 
                        for param in model.parameters():
                            param.requires_grad =True                     
                        classifier =nn.Sequential(
                                    nn.Flatten(),
                                    nn.Linear(in_features=model.classifier.in_features, out_features=1024, bias=True),
                                    nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(in_features=1024, out_features=512, bias=True),
                                    nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=512, out_features=256, bias=True),
                                    nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=256, out_features=self.N_class, bias=True),
                                    nn.LogSoftmax(dim=1) ,
                                    )
                        model.classifier = classifier 
        elif int(self.ImageShape)==self.commonShapes[2]:
            if x[0]=='efficientnet': 
                if self.model_type=='efficientnet-b2':
                    from torchvision.models import efficientnet_b2,EfficientNet_B2_Weights
                    weights=EfficientNet_B2_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.efficientnet_b2(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad =True 
                    classifier =nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(in_features=model.classifier.in_features, out_features=1408, bias=True),
                                nn.BatchNorm1d(1408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True), 
                                nn.Linear(in_features=1408, out_features=1024, bias=True),
                                nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True), 
                                nn.Linear(in_features=1024, out_features=512, bias=True),
                                nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True), 
                                nn.Linear(in_features=512, out_features=256, bias=True),
                                nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True), 
                                nn.Linear(in_features=256, out_features=2self.N_class, bias=True),
                                nn.LogSoftmax(dim=1) ,
                                )                        
                    model._fc = classifier       
        elif int(self.ImageShape)==self.commonShapes[3]:
            if x[0]=='inception' :
                from torchvision.models import inception_v3,Inception_V3_Weights
                weights=Inception_V3_Weights.IMAGENET1K_V1
                pretrained=self.pretrained
                model = models.inception_v3(weights=(weights,pretrained)).to(device)
                for param in model.parameters():
                    param.requires_grad = True
                model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, self.num_classes)
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)    
        elif int(self.ImageShape)==self.commonShapes[4]:
            if x[0]=='efficientnet':    
                if self.model_type=='efficientnet-b3':
                        from torchvision.models import efficientnet_b3,EfficientNet_B3_Weights
                        weights=EfficientNet_B3_Weights.IMAGENET1K_V1
                        pretrained=self.pretrained
                        model = models.efficientnet_b3(weights=(weights,pretrained)).to(device)
                        for param in model.parameters():
                            param.requires_grad =True
                        classifier =nn.Sequential(
                                    nn.Flatten(),
                                    nn.Linear(in_features=model.classifier.in_features, out_features=1536, bias=True),
                                    nn.BatchNorm1d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0.35, inplace=False),
                                    nn.Linear(in_features=1536, out_features=1536, bias=True),
                                    nn.BatchNorm1d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0.35, inplace=False),
                                    nn.Linear(in_features=1536, out_features=1024, bias=True),
                                    nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=0.35, inplace=False),
                                    nn.Linear(in_features=1024, out_features=512, bias=True),
                                    nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    ##nn.Dropout(p=0.50, inplace=False),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=512, out_features=256, bias=True),
                                    nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    #nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    #nn.Dropout(p=0.25, inplace=False),
                                    nn.Linear(in_features=256, out_features=self.N_class, bias=True),
                                    nn.LogSoftmax(dim=1) ,
                                    )
                        model.classifier = classifier                               
        elif int(self.ImageShape)==self.commonShapes[5]:
            if x[0]=='efficientnet':
                if self.model_type=='efficientnet-b4':
                    from torchvision.models import efficientnet_b4,EfficientNet_B4_Weights
                    weights=EfficientNet_B4_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.efficientnet_b4(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad =True
                    classifier = nn.Sequential(
                                    nn.Flatten(),
                                    nn.Linear(in_features=model.classifier.in_features, out_features=1792, bias=True),
                                    nn.BatchNorm1d(1792, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(in_features=1024, out_features=1024, bias=True),
                                    nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=1024, out_features=512, bias=True),
                                    nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=512, out_features=256, bias=True),
                                    nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=256, out_features=self.N_class, bias=True),
                                    nn.LogSoftmax(dim=1) ,
                                    )                         
                    model.classifier = classifier
        elif int(self.ImageShape)==self.commonShapes[6]:
            if x[0]=='regnet' :
                if self.model_type=='regnet_y_128gf':
                    from torchvision.models  import regnet_y_128gf, RegNet_Y_128GF_Weights
                    weights=RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1
                    pretrained=self.pretrained
                    model = models.regnet_y_128gf(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)  
                elif self.model_type=='regnet_y_16gf':
                    from torchvision.models  import regnet_y_16gf, RegNet_Y_16GF_Weights
                    weights=RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1
                    pretrained=self.pretrained
                    model = models.regnet_y_16gf(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
                elif self.model_type== 'regnet_y_32gf' :
                    from torchvision.models  import regnet_y_32gf, RegNet_Y_32GF_Weights
                    weights=RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1
                    pretrained=self.pretrained
                    model = models.regnet_y_32gf(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad = True
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            elif x[0]== 'vision_transformer':    
                if self.model_type =='vit_b_16' :
                    from torchvision.models  import vit_b_16, ViT_B_16_Weights
                    weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
                    pretrained=self.pretrained
                    model = models.vit_b_16(pretrained=self.pretrained).to(device)
                    for param in model.parameters():
                        param.requires_grad = True 
                    classifier= nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(in_features=model.fc.in_features, out_features=512, bias=True),
                                nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True), 
                                nn.Linear(in_features=512, out_features=256, bias=True),
                                nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                nn.ReLU(inplace=True), 
                                nn.Linear(in_features=256, out_features=self.num_classes, bias=True),
                                )
                    model.fc=classifier  
        elif int(self.ImageShape)==self.commonShapes[7]:
            if x[0]=='efficientnet':
                if self.model_type=='efficientnet-b5':
                    from torchvision.models import efficientnet_b5,EfficientNet_B5_Weights
                    weights=EfficientNet_B5_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.efficientnet_b5(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad =True  
                    classifier = nn.Sequential(
                                    nn.Flatten(),
                                    nn.Linear(in_features=model.classifier.in_features, out_features=2048, bias=True),
                                    nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(in_features=model.classifier.in_features, out_features=1024, bias=True),
                                    nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=1024, out_features=512, bias=True),
                                    nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=512, out_features=256, bias=True),
                                    nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=256, out_features=self.N_class, bias=True),
                                    nn.LogSoftmax(dim=1) ,
                                    )
                    model.classifier = classifier
        elif int(self.ImageShape)==self.commonShapes[8]:
            if x[0]== 'vision_transformer':    
                if self.model_type == 'vit_h_14' :
                        from torchvision.models  import vit_h_14, ViT_H_14_Weights
                        weights=ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
                        pretrained=self.pretrained
                        model = models.vit_h_14(weights=(weights,pretrained)).to(device)    
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
        elif int(self.ImageShape)==self.commonShapes[9]:
            if x[0]== 'vision_transformer':    
                if self.model_type == 'vit_l_16' :
                    from torchvision.models  import vit_l_16, ViT_L_16_Weights
                    weights=ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1
                    pretrained=self.pretrained
                    model = models.vit_l_16(weights=(weights,pretrained)).to(device)    
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
        elif int(self.ImageShape)==self.commonShapes[10]:
            if x[0]=='efficientnet':
                if self.model_type=='efficientnet-b6':
                    from torchvision.models import efficientnet_b6,EfficientNet_B6_Weights
                    weights=EfficientNet_B6_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.efficientnet_b6(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad =True                     
                    classifier = nn.Sequential(
                                    nn.Flatten(),
                                    nn.Linear(in_features=model.classifier.in_features, out_features=1024, bias=True),
                                    nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=1024, out_features=512, bias=True),
                                    nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=512, out_features=256, bias=True),
                                    nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=256, out_features=self.N_class, bias=True),
                                    nn.LogSoftmax(dim=1) ,
                                    )
                    model.classifier = classifier
                    model.to(device)
        elif int(self.ImageShape)==self.commonShapes[11]:
            if x[0]=='efficientnet':
                if self.model_type=='efficientnet-b7':
                    from torchvision.models import efficientnet_b7,EfficientNet_B7_Weights
                    weights=EfficientNet_B7_Weights.IMAGENET1K_V1
                    pretrained=self.pretrained
                    model = models.efficientnet_b7(weights=(weights,pretrained)).to(device)
                    for param in model.parameters():
                        param.requires_grad =True  
                    classifier =nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(in_features=model.classifier.in_features, out_features=1024, bias=True),
                                    nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=1024, out_features=512, bias=True),
                                    nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=512, out_features=256, bias=True),
                                    nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True), 
                                    nn.Linear(in_features=256, out_features=self.N_class, bias=True),
                                    nn.LogSoftmax(dim=1) ,
                                    )
                    model.classifier = classifier
                
        else:
            raise AssertionError(f'ImageShape "{self.ImageShape}" is not a valid size for an image !,plase insert a Valid from : {commonShapes} more info check https://medium.com/analytics-vidhya/how-to-pick-the-optimal-image-size-for-training-convolution-neural-network-65702b880f05')
        model.to(device)
        return model
    
    def optimizer(self,model):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        criterion = nn.CrossEntropyLoss()
        if int(self.ImageShape) in self.commonShapes[2:] :
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        elif int(self.ImageShape)==self.commonShapes[1]:
            optimizer=torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
            # Decay LR by a factor of 0.1 every 7 epochs
            exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        elif int(self.ImageShape)==self.commonShapes[0] :
            if self.model_type=='swin_transformer':
                import timm
                from timm.loss import LabelSmoothingCrossEntropy
                criterion = LabelSmoothingCrossEntropy()
                criterion = criterion.to(device)
                optimizer = optim.Adam(model.head.parameters(), lr=0.001)
            else:
                optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
                exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience =  5, mode = 'max', verbose=True)       
                #criterion = nn.NLLLoss() 
        
        else:
           optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
           exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience =  5, mode = 'max', verbose=True)        
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
    

