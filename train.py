 #usage example: python train.py flowers --learning_rate 0.001 --hidden_units 256 --epochs 4 --arch "vgg16" --save_dir "vgg_cp" -gpu
 #usage example: python train.py flowers --learning_rate 0.001 --hidden_units 512 --epochs 4 --arch "resnet50" --save_dir "res_cp" -gpu
 #Author: Rajnesh Kathuria

from get_input_args import get_input_args
import numpy as np
from utility import set_std_dict, set_model, save_checkpoint
from training import train_model, test_model
import json

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

in_arg = get_input_args()
std_dict = set_std_dict(in_arg.data_dir)
model = set_model(in_arg.arch, in_arg.gpu, in_arg.hidden_units, std_dict)

epochs = in_arg.epochs
learning_rate = in_arg.learning_rate

criterion = nn.NLLLoss()
if in_arg.arch == "vgg16":
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
else:
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

train_model(model, criterion, optimizer, epochs, in_arg.gpu, std_dict)
test_model(model, in_arg.gpu, std_dict)

save_checkpoint(model, in_arg.arch, optimizer, epochs, in_arg.save_dir, std_dict)
