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
   [2x1024]    -> Unflatten -> [1x1024] ->  Unflatten -> [16x8x8] ->  Transposed Conv(stride = 2) -> [8x15x15] -> Transposed  Conv + Upsampling -> [4x30x30]  -> Transposed Conv + Upsampling -> [3x60x60]
    
## Structure of Projet

FieldRoadClassifier/
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
    

