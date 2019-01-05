# image-classifier
This python program uses pyTorch and allows users to select either vgg16 or resnet50 algorithms to train a network to classify images. 
run python train.py --help to view all command line arguments for the training process
run python predict.py --help to view all command line arguments for predicting the classification of the image

The training program assumes standard pyTorch data folder structure: 

data/train/x/img1.jpg
data/train/x/img2.jpg
data/train/y/img3.jpg
..
..
data/valid/x/img4.jpg
..
data/test/x/img5.jpg
..
..
x and y are the classifications in the example below. You can use upload a json file while using predict.py 
to map classifications to actual class names
