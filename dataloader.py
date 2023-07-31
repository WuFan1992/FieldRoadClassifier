import torch
import torchvision.transforms as transforms
import random
import cv2
from configparser import ConfigParser

import os

config_parser = ConfigParser()
config_parser.read("./config/config.ini")
training_info = config_parser["TRAIN"]



class DataLoader():
    def __init__(self, data_dir="./dataset/"):
        self.data_dir = data_dir
        self.test_dir = self.data_dir + "test_images/"
        self.data_list = []
        self.train_data_list=[]
        self.test_list = []
        self.test_labels = []

    def get_files_names(self):
        field_dir = os.path.join(self.data_dir,  "fields")
        road_dir = self.data_dir + "roads"
        for filename in os.listdir(field_dir):
            self.data_list.append({"img": os.path.join(field_dir, filename), "label": torch.tensor([0,1], dtype=float)})   # example {"img": "./dataset/fields, "label":[0,1]}

        for filename in os.listdir(road_dir):
            self.data_list.append({"img": os.path.join(road_dir, filename), "label": torch.tensor([1,0], dtype=float)})

        random.shuffle(self.data_list)   # shuffle the data
        self.train_data_list = random.sample(self.data_list, int(training_info["batch_size"])) # randomly select 32 sample as training sample 

    def get_test_file(self):
        test_dir = self.data_dir + "test_images"
        for filename in os.listdir(test_dir):
            self.test_list.append(os.path.join(test_dir, filename))

    def load_img(self, index):
        img = cv2.imread(self.train_data_list[index]["img"])
        img = cv2.resize(img, (60,60), interpolation= cv2.INTER_LINEAR)  #resize the training image to (60,60), this is the input size of autoencoder_fr 
        trans = transforms.ToTensor()    # numpy to tensor format
        img = trans(img)
        return img, self.train_data_list[index]["label"]

    def load_test_img(self, index):
        img =  cv2.imread(self.test_list[index])
        img = cv2.resize(img, (60,60), interpolation= cv2.INTER_LINEAR)
        trans = transforms.ToTensor()
        img = trans(img)
        return img , self.test_labels[index]
