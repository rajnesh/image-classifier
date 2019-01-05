#Author: Rajnesh Kathuria
import argparse
import os
from os import path

def get_input_args():
    
    parser = argparse.ArgumentParser(description='Enter data folder and, optionally, checkpoint folder, architecture, learning rate, epochs, hidden units, gpu')
    parser.add_argument("data_dir", help="Required argument: Data Folder Pathname")
    parser.add_argument("--save_dir", default="vgg_cp", help="Checkpoint Folder Pathname")
    parser.add_argument("--arch", default="vgg16", help="torchvision model: vgg16 or resnet50 (default: vgg16)")
    parser.add_argument("--learning_rate", default=0.001, help="Model Learning Rate (default: 0.001)", type=float)
    parser.add_argument("--hidden_units", default=256, help="Number of Hidden Units in Model (default: 256)", type=int)
    parser.add_argument("--epochs", default=4, help="Number of Epochs to run (default: 4)", type=int)
    parser.add_argument("-gpu", help="will run on GPU if this argument is set", action="store_true")
    
    args = parser.parse_args()
  
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    return args
