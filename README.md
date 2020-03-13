# SCENES CLASSIFIER

Repo contains my own pytorch implementation of scenes classifier based on CNN architecture with variety of techniques used such as data and batch normalization, random cropping etc. 

**Architecture description**  
    *Full description is available in ./core/model.py*  
    Network takes tensor of shape 3x64x64 as input and produces 1x6 vector, which argmax represents class with highest probability  
    NN consists of five (3x3) convolution layers with batch normalization and dropout, and two plain layers.  
    Every layer's activation function is ReLU.   

**Repository description**  
  Repo contains classes and functions by means of which one can retrain network on different dataset and use existing model to train further as well as make predictions 

**Dataset**  
[Intel Image Classification](https://www.kaggle.com/puneet6060/intel-image-classification)  
  Provided dataset is split into 3 parts - training, test and prediction.  
  To make use of this dataset as well as defined csv files unzip it in ./dataset directory

**Results**  
  After 50 epochs of training with learning rate equal to 0.0007 network reached 96.7% accuracy on test dataset

![alt text](https://i.ibb.co/JdLGcwT/loss-2.png "Loss")  

![alt text](https://i.ibb.co/hLC2NYw/accuracy-1.png "Accuracy")

**Usage**  
  One can download model trained by me from [here](https://drive.google.com/open?id=1-W7ngLSqIGHkIU8kvBTlO04XSrNr1udp) and put it into ./saved_model directory


Currently there are three user interfaces implemented. Hit -h for any to see possible options
>train_network.py    
>> *allows to retrain existing model or train from beginning. Config is located in ./cfg/training.json*

>print_labeled.py
>> *prints images and their labels from csv file* 

>predict_csv.py
>> *predicts labels of images with paths stored in csv file and saves them in another csv* 

