import sys
sys.path.append('versatile_diffusion')
import os
import numpy as np
import torch
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from torch.utils.data import DataLoader, Dataset
from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
import matplotlib.pyplot as plt
import torchvision.transforms as T
import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)


##### FUNCTIONS
def compute_cliptext_embeddings(annotations, num_embed, num_features, num_rows):
    clips = np.zeros((num_rows, num_embed, num_features))
    with torch.no_grad():
        for i,annots in enumerate(annotations):
            cin = list(annots[annots!=''])
            print(i)
            c = net.clip_encode_text(cin)
            clips[i] = c.to('cpu').numpy().mean(0)
    return clips



##### SET PATHS
base_path = '' 
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



##### LOAD CAPTIONS
train_annots = np.load(processed_dir + 'subj_{}/annotations_train.npy'.format(sub)) 
test_annots = np.load(processed_dir + 'subj_{}/annotations_test.npy'.format(sub))  



##### COMPUTE CLIP EMBEDDINGS
num_embed, num_features, num_test, num_train = 77, 768, len(test_annots), len(train_annots)

train_clips = compute_cliptext_embeddings(train_annots, num_embed, num_features, num_train)
test_clips = compute_cliptext_embeddings(test_annots, num_embed, num_features, num_test)



##### SAVE
if not os.path.exists(extracted_dir + 'subj_{}/'.format(sub)):
    os.makedirs(extracted_dir + 'subj_{}/'.format(sub))
np.save(extracted_dir + 'subj_{}/cliptext_train.npy'.format(sub),  train_clips)
np.save(extracted_dir + 'subj_{}/cliptext_test.npy'.format(sub),  test_clips)





