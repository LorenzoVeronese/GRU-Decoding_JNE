import sys
sys.path.append('versatile_diffusion')
import os
import PIL
from PIL import Image
import numpy as np
import torch
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.experiments.sd_default import color_adjust, auto_merge_imlist
from torch.utils.data import DataLoader, Dataset
from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
import torchvision.transforms as T
import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)



##### FUNCTIONS
def compute_clipvision_embeddings(loader, num_embed, num_features, num_rows):
    clips = np.zeros((num_rows, num_embed, num_features))
    
    with torch.no_grad():
        for i,cin in enumerate(loader):
            print(i)
            #ctemp = cin*2 - 1
            c = net.clip_encode_vision(cin)
            clips[i] = c[0].cpu().numpy()
    return clips



##### SET PATHS
base_path = '' #'/Projects/brain-diffuser/' #/home/lveronese
processed_dir = base_path + 'data/04_processed_data/'
extracted_dir = base_path + 'data/05_extracted_features/'



##### LOAD PRETRAINED MODELS
cfgm_name = 'vd_noema'
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.clip = net.clip.to(device)



##### DEFINE CLASS FOR DATASET ITERATION
class batch_generator_external_images(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.im = np.load(data_path).astype(np.uint8)

    def __getitem__(self, idx):
        img = Image.fromarray(self.im[idx])
        img = T.functional.resize(img,(512,512))
        img = T.functional.to_tensor(img).float()
        #img = img/255
        img = img*2 - 1
        return img

    def __len__(self):
        return  len(self.im)
    
    
    
##### LOAD IMAGE
image_path = processed_dir + 'subj_{}/stimuli_train.npy'.format(sub)
train_images = batch_generator_external_images(data_path = image_path)

image_path = processed_dir + 'subj_{}/stimuli_test.npy'.format(sub)
test_images = batch_generator_external_images(data_path = image_path)

batch_size=1
train_loader = DataLoader(train_images, batch_size, shuffle=False)
test_loader = DataLoader(test_images, batch_size, shuffle=False)



##### COMPUTE CLIP EMBEDDINGS
num_embed, num_features, num_test, num_train = 257, 768, len(test_images), len(train_images)

train_clips = compute_clipvision_embeddings(train_loader, num_embed, num_features, num_train)
test_clips = compute_clipvision_embeddings(test_loader, num_embed, num_features, num_test)



##### SAVE
if not os.path.exists(extracted_dir + 'subj_{}/'.format(sub)):
    os.makedirs(extracted_dir + 'subj_{}/'.format(sub))
np.save(extracted_dir + 'subj_{}/clipvision_train.npy'.format(sub),  train_clips)
np.save(extracted_dir + 'subj_{}/clipvision_test.npy'.format(sub),  test_clips)






    
