#Author: Rajnesh Kathuria
import numpy as np
import json

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

from PIL import Image

imageNet_means = [0.485, 0.456, 0.406]
imageNet_SDs = [0.229, 0.224, 0.225]

def load_and_rebuild(path, gpu):
    checkpoint = torch.load(path)
    arch = checkpoint['arch']
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier = checkpoint['classifier']
    else:    
        model = models.resnet50(pretrained=True)
        model.fc = checkpoint['fc']

    for param in model.parameters():
        param.requires_grad = False
        
    
    model.load_state_dict(checkpoint['state_dict'])
    if gpu:
        model.to('cuda')
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    model.epochs = checkpoint['epochs']

    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

def process_image(image_file):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image_f = Image.open(image_file)
    
    #resize the PIL image: first check ratio of width to height and resize
    width = image_f.size[0]
    height =  image_f.size[1]
    
    image_ratio = width / height
    
    if image_ratio > 1:
        image_f.thumbnail((int(width * image_ratio), 256))
    else:
        image_f.thumbnail((256, int(height * image_ratio)))

    #center crop 224x224 e.g. left crop = (256 - 224) / 2 = 16
    image = image_f.crop((16, 16, 240, 240))
    
    #convert to np array
    image = np.array(image)
    
    #transpose order of color channel for PyTorch (moving from 3rd dim to 1st dim)
    image = image.transpose((2,0,1))
    
    #convert channels to float numbers between 0 and 1
    image = image / 255
    
    #subtract the means from each color channel (normalization)
    means = np.array(imageNet_means).reshape((3, 1, 1))
    image = image - means
    
    #divide the mean-subtracted color channel by the SDs (normalization)
    SDs = np.array(imageNet_SDs).reshape((3, 1, 1))
    image = image / SDs
    
    return image

def predict(image_path, gpu, model, topk=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    top_classes = []
    actual_class = "not in test"
    #actual_class = image_path.split('/')[-2] #if testing, uncomment to capture actual class name

    image = torch.Tensor(process_image(image_path))
    if gpu:
        image = image.to('cuda')
    image = image.view(1, 3, 224, 224)

    with torch.no_grad():
        model.eval()
        output = model(image)
        probs = torch.exp(output) #since the model outputs logs of probablities, raising them to e

        topk_probs, topk_indexes = probs.topk(topk, dim=1)
        
        top_classes = [model.idx_to_class[c] for c in topk_indexes.cpu().numpy()[0]]
        top_probs = topk_probs.cpu().numpy()[0]
       
        return top_probs, top_classes, actual_class
