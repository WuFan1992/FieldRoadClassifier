

from  model.auto_encoder import AutoencoderFR,DataLoader
import torch
from configparser import ConfigParser


config_parser = ConfigParser()
config_parser.read("./config/config.ini")
test_info = config_parser["TEST"]



test_label_list=[[0,1], [1,0], [1,0], [0,1], [1,0], [1,0],[1,0], [1,0],[0,1], [0,1]]


model = AutoencoderFR()
model.load_state_dict(torch.load(test_info["pkl_path"]))
model.eval()

data = DataLoader()
data.get_test_file()
data.test_labels = test_label_list

def predict_correct(out_cl,label):
    max_value_cl = max(out_cl)
    max_value_cl_idx = out_cl.index(max_value_cl)
    max_value_label_idx = label.index(1)
    return max_value_cl_idx == max_value_label_idx

correct_sum = 0

for index in range(len(data.test_labels)):
    img, label = data.load_test_img(index)
    out, out_cl = model(img)
    out_cl = out_cl.tolist()
    if predict_correct(out_cl, label):
        correct_sum +=1
print ('correct prediction: {}/{}'.format(correct_sum, 10))







