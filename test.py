

from  model.auto_encoder import AutoencoderFR
from  dataloader import DataLoader
import torch
from configparser import ConfigParser


config_parser = ConfigParser()
config_parser.read("./config/config.ini")
test_info = config_parser["TEST"]





# the label of test images with respect to the given order 
test_label_list=[[0,1], [1,0], [1,0], [0,1], [1,0], [1,0],[1,0], [1,0],[0,1], [0,1]]  # [0,1] -> field   [1,0] -> road
"""
predict_correct : compare the output result with the label
                  the output result will be liked [0.35, 0.65] and the label 
                  will be like [0,1]
"""
def predict_correct(out_cl,label):
    max_value_cl = max(out_cl)     # find the index of max value of classification 
    max_value_cl_idx = out_cl.index(max_value_cl)   
    max_value_label_idx = label.index(1)     # find the index of max value of label
    return max_value_cl_idx == max_value_label_idx   # these two index need to be the same if predict correctly 


if __name__ == '__main__':
  model = AutoencoderFR()
  model.load_state_dict(torch.load(test_info["pkl_path"]))
  model.eval()

  data = DataLoader()
  data.get_test_file()
  data.test_labels = test_label_list

  correct_sum = 0

  for index in range(len(data.test_labels)):
      img, label = data.load_test_img(index)
      out, out_cl = model(img)
      out_cl = out_cl.tolist()
      if predict_correct(out_cl, label):
          correct_sum +=1
  print ('correct prediction: {}/{}'.format(correct_sum, 10))







