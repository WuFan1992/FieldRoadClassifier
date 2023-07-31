
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import random
from configparser import ConfigParser
from model.auto_encoder import AutoencoderFR
from dataloader import DataLoader
from configparser import ConfigParser


config_parser = ConfigParser()
config_parser.read("./config/config.ini")
training_info = config_parser["TRAIN"]

if __name__ == '__main__':
  model = AutoencoderFR()
  distance   = nn.MSELoss()
  class_loss = nn.CrossEntropyLoss()

  optimizer = torch.optim.SGD(model.parameters(), lr=float(training_info["learning_rate"]), momentum=0.5)

  mse_multp = 0.5   
  cls_multp = 0.5
 
  model.train()

  dataloader = DataLoader(data_dir=training_info["data_dir"])
  dataloader.get_files_names()

  for epoch in range(int(training_info["num_epochs"])):
      total_mseloss = 0.0
      total_clsloss = 0.0
      for index in range(len(dataloader.train_data_list)):
          img, labels = dataloader.load_img(index) 
          output, output_cl = model(img)
          output = torch.squeeze(output)  # delete the batch size dimension
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
    
      print('epoch [{}/{}], loss_mse: {:.4f}  loss_cls: {:.4f}'.format(epoch+1, int(training_info["num_epochs"]), total_mseloss / len(dataloader.data_list), total_clsloss / len(dataloader.data_list)))
  torch.save(model.state_dict(), "./cache/fr_classifier.pkl")