# image-classifier
This python program uses pyTorch and allows users to select either vgg16 or resnet50 algorithms to train a network to classify images on a GPU (GPU can be specified using -gpu parameter on the command line). 

run "python train.py --help" to view all command line arguments for the training process

run "python predict.py --help" to view all command line arguments for predicting the classification of the image


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

x and y are classifications in the example below. You can upload a json file while using predict.py 
to map classifications to actual class names

Example usages:

 python train.py flowers --learning_rate 0.001 --hidden_units 256 --epochs 4 --arch "vgg16" --save_dir "vgg_cp" -gpu
 
 python train.py flowers --learning_rate 0.001 --hidden_units 512 --epochs 4 --arch "resnet50" --save_dir "res_cp" -gpu
 
 python predict.py flowers/test/1/image_06743.jpg vgg_cp/checkpoint.pth -gpu --top_k 2
 
