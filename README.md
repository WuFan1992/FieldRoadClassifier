# FieldRoadClassifier
Field and Road classifier using autoencoder 


## Projet Introduction

This project is based on the challenge to classify the road and field with a very tiny training set (45 images for field and 108 images for road). In this condition, deeper model, which present the higer capacity,  will be ignored because of the lack of data, probably leading to underfitting. Some model which works for the tiny data set is to be considered such as Unet. Here we apply the autoencoder because autoencoder can transform the data into lower dimension for the representation using encoder and the reproduction of artificial image using decoder will help the classification. 

## Model Introduction 
   Dataset images are in differents sizes, all the images will be resized to (60,60) before sent into the autoencoder.
   Encoder:
   [C x H x W]
   [3x60x60] -> Conv -> [4x60x60] -> Maxpooling -> [4x30x30] -> Conv -> [8x30x30] -> Maxpooling -> [8x15x15] -> Conv (stride = 2) -> [16x8x8]  -> Flatten -> [1x1024] -> fully-connected -> [1024x2]
   
   Decoder
   [2x1024]    -> Conv -> [1x1024] ->  Unflatten -> [16x8x8] ->  Transposed Conv(stride = 2) -> [8x15x15] -> Transposed  Conv + Upsampling -> [4x30x30]  -> Transposed Conv + Upsampling -> [3x60x60]
    
## Structure of Projet


    dataset/
        fields/
           1.jpg
           2.jpg
           ...
        roads/
           1.jpg
           2.jpg
           ...
        test_images/
           1.jpg
           ...
    cache/
        fr_classifier.pkl
    config/
        config.ini  
    model/
        auto_encoder.py
        __init__.py 
    dataloader.py
    test.py
    train.py
    

A pretrained model can be found in /cache/ which give the same results as the following "Results" parts shows 

## Set Environment
Make sure that you download the latest version of pytorch (at least version 1.20)
'pip3 install pytorch'

The following hyperparameters can be modified inside the ./config/config.ini : learning rate, data_dir number of epochs and batch size  

## Training and Testing
In the projet folder run  'python3 train.py' and 'python3 test.py'

## Results
Different from the traditional training with each epoch containing all the dataset (seperated with several batch), Here we define 100 epochs and randomly selected  32 images for each epoch, A better score will be obtained

After 100 epoch, generative loss (MSE) is 0.0143 and the classification loss is 0.0751. 
Using the test data set, we get 8/10 correct prediction (80%) 

# Go further 
In this projet, we do not use the data augmentation, this methods can be applied in the future to have a better results.    

