import torch
from models.resnet import *
from models.edgenext import *
from models.yolov5 import *
from models.squeezenet import *
from models.mobilenetv2 import MobileNetV2
from models.efficientnet import EfficientNetLite
from models.mobilenetv3 import mobilenetv3
from models.edgenext import create_edgenext_xx_small
from facenet_pytorch import InceptionResnetV1
from models.convnext import *
import argparse
import time
import os
import numpy as np
from timm.models import create_model
from efficientnet_pytorch import EfficientNet
import timm
import pdb

def model_build(model_name:str, num_classes:int, in_channel:int):
    """
    <args>
        model_name : from config/@.yaml
        resume:
            if '' : no pretrained model, no resume
            else : use .ckpt file and resume the model
    """

    model_name = model_name.lower()

    ######      build model     ######
    # define class or def for building model at each [@ Net].py
    # just add if ~ : ~ code for another model like below.
    #

    if model_name == 'resnet18':
        model = ResNet(size=18, num_classes=num_classes)
    
    if model_name == 'resnet34':
        model = ResNet(size=34, num_classes=num_classes)
    
    if model_name == 'resnet50':
        model = ResNet(size=50, num_classes=num_classes)
    
    if model_name == 'resnext50_4':
        model = ResNext(size=50, ext=4, num_classes=num_classes)
    
    if model_name == 'resnext101_8':
        model = ResNext(size=101, ext=8, num_classes=num_classes)
    
    if model_name == 'resnext101_4':
        model = ResNext(size=101, ext=4, num_classes=num_classes)

    if model_name == 'edgenext':
        model = create_edgenext_xx_small(num_classes=num_classes)
    
    if model_name == 'efficientnetlite':
        model = EfficientNetLite(num_classes=num_classes)

    if model_name == 'mobilenetv2_0.25':
        model = MobileNetV2(num_classes=num_classes, width_mult=0.25)
    
    if model_name == 'mobilenetv2_0.5':
        model = MobileNetV2(num_classes=num_classes, width_mult=0.5)

    if model_name == 'yolov5n':
        try:
            model = Model('yolov5n.yaml')
        except:
            model = Model('models/yolov5n.yaml')
            
    if model_name == 'squeezenet1_1':
        model = squeezenet1_1(num_classes=num_classes)
        
    if model_name == 'mobilenetv3_0.5':
        model = mobilenetv3(num_classes=num_classes, width_mult=0.5)

    if model_name == 'mobilenetv3_0.75':
        model = mobilenetv3(num_classes=num_classes, width_mult=0.75)

    if model_name == 'inceptionresnet':
        model = InceptionResnetV1(pretrained='vggface2', classify=True)
        model.logits = nn.Linear(512, num_classes, bias=True)

    if model_name == 'b0':
        model = EfficientNet.from_pretrained('efficientnet-b0')
        model._fc = nn.Linear(1280, num_classes, bias=True)
    
    if model_name == 'b1':
        model = EfficientNet.from_pretrained('efficientnet-b1')
        model._fc = nn.Linear(1280, num_classes, bias=True)

    if model_name == 'b2':
        model = EfficientNet.from_pretrained('efficientnet-b2')
        model._fc = nn.Linear(1408, num_classes, bias=True)

    if model_name == 'b3':
        model = EfficientNet.from_pretrained('efficientnet-b3')
        model._fc = nn.Linear(1280, num_classes, bias=True)

    if model_name == 'b4':
        model = EfficientNet.from_pretrained('efficientnet-b4')
        model._fc = nn.Linear(1792, num_classes, bias=True)
    
    if model_name == 'b5':
        model = EfficientNet.from_pretrained('efficientnet-b5')
        model._fc = nn.Linear(1280, num_classes, bias=True)

    if model_name == 'b6':
        model = EfficientNet.from_pretrained('efficientnet-b6')
        model._fc = nn.Linear(1280, num_classes, bias=True)
    
    if model_name == 'b7':
        model = EfficientNet.from_pretrained('efficientnet-b7')
        model._fc = nn.Linear(1280, num_classes, bias=True)

    if model_name == 'convnext_tiny':
        model = convnext_tiny(pretrained=True)
        model.head = nn.Linear(768, num_classes, bias=True)
        
    if model_name == 'convnext_small':
        model = convnext_small(pretrained=True)
        model.head = nn.Linear(768, num_classes, bias=True)
        
    if model_name == 'convnext_base':
        model = convnext_base(pretrained=True)
        model.head = nn.Linear(1024, num_classes, bias=True)
        
    return model


