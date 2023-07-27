
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image


import os

num_epochs = 10
batch_size = 64
transform = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))
 ])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DataLoader():
    def __init__(self, data_dir="../dataset/"):
        self.data_dir = data_dir
        self.data_list = []
        self.label_list = []

    def _get_files_names(self):
        field_dir = self.data_dir + "fileds"
        road_dir = self.data_dir + "roads"
        for filename in os.listdir(field_dir):
            self.data_list.append(os.path.join(field_dir, filename))
            self.label_list.append(0)
        for filename in os.listdir(road_dir):
            self.label_list.append(os.path.join(road_dir, filename))
            self.label_list.append(1)

    def load_img(self, index):
        return self.data_list[index], self.label_list[index]


def getTrainData(data_path="../data/"):
     
    



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
             nn.Flatten(),
             nn.Linear(1024, 2)
             )
         self.softmax = nn.Softmax(dim=1)
         self.decoder = nn.Sequential(
             nn.Linear(2, 1024),
             nn.ReLU(True),
             nn.Unflatten(1, (16, 8, 8)),
             nn.Dropout2d(p=0.1),
             nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=1),
             nn.ReLU(True),
             nn.Upsample(scale_factor=2, mode='bilinear'),
             nn.Dropout2d(p=0.1),
             nn.ConvTranspose2d(8, 4, kernel_size=3, stride=1, padding=1), 
             nn.ReLU(True),
             nn.Upsample(scale_factor=2),
             nn.Dropout2d(p=0.1),
             nn.Conv2d(4,3, kernel_size=3, stride=1, padding=1)
             )
         
     def forward(self, x):
         out_en = self.encoder(x)
         out = self.softmax(out_en)
         out = self.decoder(out)
         return out, out_en
     
model = AutoencoderFR().to(device)
distance   = nn.MSELoss()
class_loss = nn.CrossEntropyLoss()
 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
 
mse_multp = 0.5
cls_multp = 0.5
 
model.train()
 
for epoch in range(num_epochs):
    total_mseloss = 0.0
    total_clsloss = 0.0
    for ind, data in enumerate(dataloader):
        img, labels = data[0].to(device), data[1].to(device) 
        output, output_en = model(img)
        loss_mse = distance(output, img)
        loss_cls = class_loss(output_en, labels)
        loss = (mse_multp * loss_mse) + (cls_multp * loss_cls)  # Combine two losses together
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Track this epoch's loss
        total_mseloss += loss_mse.item()
        total_clsloss += loss_cls.item()
 
     # Check accuracy on test set after each epoch:
    model.eval()   # Turn off dropout in evaluation mode
    acc = 0.0
    total_samples = 0
    for data in testloader:
        # We only care about the 10 dimensional encoder output for classification
        img, labels = data[0].to(device), data[1].to(device) 
        _, output_en = model(img)   
        # output_en contains 10 values for each input, apply softmax to calculate class probabilities
        prob = nn.functional.softmax(output_en, dim = 1)
        pred = torch.max(prob, dim=1)[1].detach().cpu().numpy() # Max prob assigned to class 
        acc += (pred == labels.cpu().numpy()).sum()
        total_samples += labels.shape[0]
    model.train()   # Enables dropout back again
    print('epoch [{}/{}], loss_mse: {:.4f}  loss_cls: {:.4f}  Acc on test: {:.4f}'.format(epoch+1, num_epochs, total_mseloss / len(dataloader), total_clsloss / len(dataloader), acc / total_samples))

