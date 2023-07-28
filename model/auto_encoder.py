
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

import cv2


import os

num_epochs = 10
batch_size = 64
transform = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))
 ])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DataLoader():
    def __init__(self, data_dir="./dataset/"):
        self.data_dir = data_dir
        self.data_list = []
        self.label_list = []
        self.test_list = []

    def _get_files_names(self):
        field_dir = os.path.join(self.data_dir,  "fields")
        road_dir = self.data_dir + "roads"
        for filename in os.listdir(field_dir):
            self.data_list.append(os.path.join(field_dir, filename))
            self.label_list.append(torch.tensor([0,1], dtype=float))
        for filename in os.listdir(road_dir):
            self.label_list.append(os.path.join(road_dir, filename))
            self.label_list.append(torch.tensor([1,0], dtype=float))

    def get_test_file(self):
        test_dir = self.data_dir + "test_images"
        for filename in os.listdir(test_dir):
            self.test_list.append(os.path.join(test_dir, filename))

    def load_img(self, index):
        img = cv2.imread(self.data_list[index])
        img = cv2.resize(img, (60,60), interpolation= cv2.INTER_LINEAR)
        trans = transforms.ToTensor()
        img = trans(img)
        return img, self.label_list[index]

    def load_test_img(self, name):
        img =  cv2.imread(name)
        img = img.transpose((2, 0, 1))
        img = cv2.resize(img, (60,60))
        return img

    
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
             nn.Flatten(0,-1), 
             nn.Linear(1024, 2)
             )
         #self.softmax = torch.nn.functional.softmax()
         self.linear_decoder = nn.Linear(2,1024)
         self.decoder = nn.Sequential(
             #nn.Linear(2, 1024),
             nn.ReLU(True),
             nn.Unflatten(1, (16, 8, 8)),
             nn.Dropout2d(p=0.1),
             nn.ReLU(True),
             nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1),
             nn.Dropout2d(p=0.1),
             nn.ReLU(True),
             nn.ConvTranspose2d(8, 4, kernel_size=3, stride=1, padding=1), 
             nn.Upsample(scale_factor=2),
             nn.Dropout2d(p=0.1),
             nn.ReLU(True),
             nn.ConvTranspose2d(4,3, kernel_size=3, stride=1, padding=1),
             nn.Upsample(scale_factor=2),
             )
         
     def forward(self, x):
         out_en = self.encoder(x)
         out_cl = torch.softmax(out_en, dim=0)
         out = self.linear_decoder(out_en)
         out = torch.unsqueeze(out,0)
         out = self.decoder(out)
         return out, out_cl
     
model = AutoencoderFR()#.to(device)
distance   = nn.MSELoss()
class_loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
 
mse_multp = 0.5
cls_multp = 0.5
 
model.train()

dataloader = DataLoader()
dataloader._get_files_names()

for epoch in range(num_epochs):
    total_mseloss = 0.0
    total_clsloss = 0.0
    for index in range(len(dataloader.data_list)):
        img, labels = dataloader.load_img(index) 
        output, output_cl = model(img)
        output = torch.squeeze(output)
        loss_mse = distance(output, img)
        loss_cls = class_loss(output_cl, labels)
        loss = (mse_multp * loss_mse) + (cls_multp * loss_cls)  # Combine two losses together
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Track this epoch's loss
        total_mseloss += loss_mse.item()
        total_clsloss += loss_cls.item()
 
     # Check accuracy on test set after each epoch:
    
    print('epoch [{}/{}], loss_mse: {:.4f}  loss_cls: {:.4f}'.format(epoch+1, num_epochs, total_mseloss / len(dataloader.data_list), total_clsloss / len(dataloader.data_list)))




"""

model.eval()   # Turn off dropout in evaluation mode
    acc = 0.0
    total_samples = 0
    for name in dataloader.test_list:
        # We only care about the 10 dimensional encoder output for classification
        img, labels = data[0].to(device), data[1].to(device) 
        _, output_en = model(img)   
        # output_en contains 10 values for each input, apply softmax to calculate class probabilities
        prob = nn.functional.softmax(output_en, dim = 1)
        pred = torch.max(prob, dim=1)[1].detach().cpu().numpy() # Max prob assigned to class 
        acc += (pred == labels.cpu().numpy()).sum()
        total_samples += labels.shape[0]
    model.train()   # Enables dropout back again


"""