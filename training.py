#Author Rajnesh Kathuria
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def train_model(model, criterion, optimizer, epochs, gpu, std_dict):
    for e in range(epochs):     
        for train_val in ['train','valid']:
            running_loss = 0
            running_accuracy = 0
            avg_loss_in_epoch = 0
            avg_accuracy_in_epoch = 0
        
            if train_val == 'train':
                model.train(True)
                print ("Epoch {}/{}: in training".format(e+1, epochs))
            else:
                model.train(False)
                model.eval()
                print ("Epoch {}/{}: in validation".format(e+1, epochs))
                
            for inputs, labels in std_dict['dataloaders'][train_val]:
                if gpu:
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')
                optimizer.zero_grad()
            
                # Forward and backward passes
                outputs = model.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_accuracy += (predicted == labels).sum().item()
                        
            avg_loss_in_epoch = running_loss / std_dict['dataset_size'][train_val]
            avg_accuracy_in_epoch = 100 * running_accuracy / std_dict['dataset_size'][train_val]            
            print("Epoch {} results: ".format(train_val), "Average Loss: {:.4f}".format(avg_loss_in_epoch), 
                        "Average Accuracy: {:.1f} %".format(avg_accuracy_in_epoch))

def test_model(model, gpu, std_dict):
    correct = 0

    with torch.no_grad():
        for images, labels in std_dict['dataloaders']['test']:
            if gpu:
                images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    print('Number of test images: {}'.format(std_dict['dataset_size']['test']))
    print('Accuracy of the network on test images: %d %%' % (100 * correct / std_dict['dataset_size']['test']))
