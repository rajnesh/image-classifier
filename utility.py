import torch
import os
from torch import nn, optim
from torchvision import datasets, transforms, models

def set_std_dict(data_dir):
    std_dict = {}
    std_dict['train_dir'] = data_dir + '/train'
    std_dict['valid_dir'] = data_dir + '/valid'
    std_dict['test_dir']= data_dir + '/test'

    std_dict['imageNet_means'] = [0.485, 0.456, 0.406]
    std_dict['imageNet_SDs'] = [0.229, 0.224, 0.225]
    std_dict['train_data_batch_size'] = 128
    std_dict['valid_data_batch_size'] = 64
    std_dict['test_data_batch_size'] = 64

    std_dict['data_transforms'] = {
        'train': 
            transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(std_dict['imageNet_means'], std_dict['imageNet_SDs'])]),
        'valid': 
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(std_dict['imageNet_means'], std_dict['imageNet_SDs'])]),
        'test': 
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(std_dict['imageNet_means'], std_dict['imageNet_SDs'])])
    }

    std_dict['image_datasets'] = {
        'train': datasets.ImageFolder(std_dict['train_dir'], transform=std_dict['data_transforms']['train']),
        'valid': datasets.ImageFolder(std_dict['valid_dir'], transform=std_dict['data_transforms']['valid']),
        'test': datasets.ImageFolder(std_dict['test_dir'], transform=std_dict['data_transforms']['test'])
    }

    std_dict['dataloaders'] = {
        'train': torch.utils.data.DataLoader(std_dict['image_datasets']['train'], batch_size=std_dict['train_data_batch_size'], shuffle=True),
        'valid': torch.utils.data.DataLoader(std_dict['image_datasets']['valid'], batch_size=std_dict['valid_data_batch_size'], shuffle=True),
        'test': torch.utils.data.DataLoader(std_dict['image_datasets']['test'], batch_size=std_dict['test_data_batch_size'], shuffle=True)
    }

    std_dict['dataset_size'] = {}
    for key, value in std_dict['image_datasets'].items():
        std_dict['dataset_size'][key] = len(value)

    std_dict['categories'] = []
    for d in os.listdir(std_dict['train_dir']):
        std_dict['categories'].append(d)
    std_dict['num_classes'] = len(std_dict['categories'])

    return std_dict

def set_model(model_name, gpu, hidden_units, std_dict):

    if model_name == "vgg16":
        model = models.vgg16(pretrained=True)
        num_inputs = model.classifier[6].in_features

        for param in model.features.parameters():
            param.requires_grad = False

        model.classifier[6] = nn.Sequential(
            nn.Linear(num_inputs, hidden_units), 
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(hidden_units, std_dict['num_classes']),                   
            nn.LogSoftmax(dim=1))

    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        num_inputs = model.fc.in_features

        for param in model.parameters():
            param.requires_grad = False
        
        model.fc = nn.Sequential(
            nn.Linear(num_inputs, hidden_units), 
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(hidden_units, std_dict['num_classes']),                   
            nn.LogSoftmax(dim=1))

    else:
        print("Cannot use model. Please select vgg16 or resnet50")
        exit()

    if gpu:
        model = model.to('cuda')

    return model


def save_checkpoint(model, arch, optimizer, epochs, save_dir,std_dict):
    checkpoint = {}
    model.class_to_idx = std_dict['image_datasets']['train'].class_to_idx
    
    model.idx_to_class = {}
    for i_class, index in model.class_to_idx.items():
        model.idx_to_class[index] = i_class
        
    checkpoint['epochs'] = epochs
    checkpoint['arch'] = arch
    checkpoint['class_to_idx'] = model.class_to_idx
    checkpoint['idx_to_class'] = model.idx_to_class
    if arch == "vgg16":
        checkpoint['classifier'] = model.classifier
    else:
        checkpoint['fc'] = model.fc
    checkpoint['state_dict'] = model.state_dict()

    checkpoint['optimizer'] = optimizer
    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, save_dir + "/checkpoint.pth")
    print('checkpoint saved to {}'.format(save_dir + "/checkpoint.pth"))