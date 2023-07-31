
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import random
import cv2
from configparser import ConfigParser

import os


config_parser = ConfigParser()
config_parser.read("./config/config.ini")
training_info = config_parser["TRAIN"]

    
""" AutoencoderFR means Autoencoder Field Road"""
class AutoencoderFR(nn.Module):
     def __init__(self):
         super(AutoencoderFR,self).__init__()
         self.encoder = nn.Sequential(
             nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
             nn.Dropout2d(p=0.1),
             nn.ReLU(True),
             nn.MaxPool2d(kernel_size=2),   # 4x30x30
             nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1 ),
             nn.Dropout2d(p=0.1),
             nn.ReLU(True),
             nn.MaxPool2d(kernel_size=2), #8x15x15
             nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1 ), #16x8x8
             nn.Dropout2d(p=0.1),
             nn.ReLU(True),
             nn.Flatten(0,-1),     # 1x1024
             nn.Linear(1024, 2)   # 1024x2
             )

         self.linear_decoder = nn.Linear(2,1024)
         self.decoder = nn.Sequential(
             nn.ReLU(True),
             nn.Unflatten(1, (16, 8, 8)),  # 1x1024
             nn.Dropout2d(p=0.1),
             nn.ReLU(True),
             nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1), # 8x15x15
             nn.Dropout2d(p=0.1),
             nn.ReLU(True),
             nn.ConvTranspose2d(8, 4, kernel_size=3, stride=1, padding=1),  # 4x30x30
             nn.Upsample(scale_factor=2),
             nn.Dropout2d(p=0.1),
             nn.ReLU(True),
             nn.ConvTranspose2d(4,3, kernel_size=3, stride=1, padding=1),  #3x60x60
             nn.Upsample(scale_factor=2),     
             )
         
     def forward(self, x):
         out_en = self.encoder(x)   # encoder 
         out_cl = torch.softmax(out_en, dim=0)  # connect encoder with softmax to output the classification result
         out = self.linear_decoder(out_en)      # fully connected layer   
         out = torch.unsqueeze(out,0)           # add batch size dimension to 1
         out = self.decoder(out)                # decoder
         return out, out_cl            
