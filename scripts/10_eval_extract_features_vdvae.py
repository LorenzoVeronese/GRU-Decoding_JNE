import os
import sys
import numpy as np
import h5py
import scipy.io as spio
import nibabel as nib
import torch
import torchvision
import torchvision.models as tvmodels
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image
import clip
import skimage.io as sio
from skimage import data, img_as_float
from skimage.transform import resize as imresize
from skimage.metrics import structural_similarity as ssim
import scipy as sp
import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-use_other_model_as_first_step", "--use_other_model_as_first_step",help="0: use vdvae as first step", default=0)
args = parser.parse_args()
sub = int(args.sub)
use_other_model_as_first_step=int(args.use_other_model_as_first_step)



##### SET PATHS
base_path = '' 
processed_dir = base_path + 'data/04_processed_data/'
extracted_dir = base_path + 'data/05_extracted_features/'
regression_dir = base_path + 'data/06_regression_weights/'
predicted_dir = base_path + 'data/07_predicted_features/'
results_vae_dir = base_path + 'results/vdvae/'
results_versatile_dir = base_path + 'results/versatile_diffusion/'
results_gt_dir = base_path + 'results/ground_truth/'
eval_features_dir = base_path + 'data/08_eval_features/'



##### SET PATHS
print()
print("##############################")
print('Setting folders:')
is_gt = 0
if sub in allowed_subjs and is_gt == 0:
    feats_dir = eval_features_dir + 'subj_{}/vdvae/'.format(sub)
    images_dir = results_vae_dir + 'subj_{}/'.format(sub)
else:
    print("is_gt==1 is not applicable to this code: use 10_eval_extract_featres.py instead")
    
if not os.path.exists(feats_dir):
   os.makedirs(feats_dir)
    
print('    Images (src) folder: ' + str(images_dir))
print('    Features (dst) folder: ' + str(feats_dir))
    


##### LOAD IMAGES
class batch_generator_external_images(Dataset):

    def __init__(self, data_path ='', prefix='', net_name='clip'):
        self.data_path = data_path
        self.prefix = prefix
        self.net_name = net_name
        
        if self.net_name == 'clip':
           self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        else:
           self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        test_images = np.load(processed_dir + 'subj_{}/stimuli_test.npy'.format(sub))
        self.num_test = len(test_images)#65#982
        
    def __getitem__(self,idx):
        img = Image.open(self.data_path + '{}{}.png'.format(self.prefix,idx))
        img = T.functional.resize(img,(224,224))
        img = T.functional.to_tensor(img).float()
        img = self.normalize(img)
        return img

    def __len__(self):
        return  self.num_test



##### SET FEATURES TO COMPUTE AND PARAMETERS
global feat_list
feat_list = []
def fn(module, inputs, outputs):
    feat_list.append(outputs.cpu().numpy())

net_list = [
    ('inceptionv3','avgpool'),
    ('clip','final'),
    ('alexnet',2),
    ('alexnet',5),
    ('efficientnet','avgpool'),
    ('swav','avgpool')
]

device = 0 #it was 1
net = None
batchsize=64



##### EXTRACT FEATURES
print()
print("##############################")
print('Computing features...')
if not os.path.exists(feats_dir):
    os.makedirs(feats_dir)
for (net_name,layer) in net_list:
    feat_list = []
    print(net_name,layer)
    dataset = batch_generator_external_images(data_path=images_dir,net_name=net_name,prefix='')
    loader = DataLoader(dataset,batchsize,shuffle=False)
    
    # Inception-V3
    if net_name == 'inceptionv3': # SD Brain uses this
        print('    Inception-V3...')
        net = tvmodels.inception_v3(pretrained=True)
        if layer== 'avgpool':
            net.avgpool.register_forward_hook(fn) 
        elif layer == 'lastconv':
            net.Mixed_7c.register_forward_hook(fn)
    # Alex-Net
    elif net_name == 'alexnet':
        print('    Alex-Net...')
        net = tvmodels.alexnet(pretrained=True)
        if layer==2:
            net.features[4].register_forward_hook(fn)
        elif layer==5:
            net.features[11].register_forward_hook(fn)
        elif layer==7:
            net.classifier[5].register_forward_hook(fn)
    # CLIP    
    elif net_name == 'clip':
        print('    CLIP...')
        model, _ = clip.load("ViT-L/14", device='cuda:{}'.format(device))
        net = model.visual
        net = net.to(torch.float32)
        if layer==7:
            net.transformer.resblocks[7].register_forward_hook(fn)
        elif layer==12:
            net.transformer.resblocks[12].register_forward_hook(fn)
        elif layer=='final':
            net.register_forward_hook(fn)
    # EfficientNet
    elif net_name == 'efficientnet':
        print('    EfficientNet..')
        net = tvmodels.efficientnet_b1(weights=True)
        net.avgpool.register_forward_hook(fn) 
    # SwAV
    elif net_name == 'swav':
        print('    SwAV...')
        net = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        net.avgpool.register_forward_hook(fn) 
    net.eval()
    net.cuda(device)    
    
    with torch.no_grad():
        for i,x in enumerate(loader):
            print(i*batchsize)
            x = x.cuda(device)
            _ = net(x)
    if net_name == 'clip':
        if layer == 7 or layer == 12:
            feat_list = np.concatenate(feat_list,axis=1).transpose((1,0,2))
        else:
            feat_list = np.concatenate(feat_list)
    else:   
        feat_list = np.concatenate(feat_list)
    
    # Save
    file_name = feats_dir + '{}_{}.npy'.format(net_name,layer)
    np.save(file_name, feat_list)
