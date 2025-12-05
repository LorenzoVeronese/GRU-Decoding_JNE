# NB: TO BE RUN WITH 'brain-diffuser' AS WORKING DIRECTORY (same for other scripts in the 'scripts' directory)
import sys
sys.path.append('vdvae')
import torch
import numpy as np
import socket
import argparse
import os
import json
import subprocess
from utils import (logger,
                   local_mpi_rank,
                   mpi_size,
                   maybe_download,
                   mpi_rank)
from data import mkdir_p
from contextlib import contextmanager
import torch.distributed as dist
from vae import VAE
from torch.nn.parallel.distributed import DistributedDataParallel
from train_helpers import restore_params
from image_utils import *
from model_utils import *
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
import pickle
import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-bs", "--bs",help="Batch Size",default=60)
args = parser.parse_args()
sub=int(args.sub)
batch_size=int(args.bs)


##### FUNCTIONS
def extract_latents(loader, num_latents=31):
    latents = []
    for i, x in enumerate(loader):
      data_input, target = preprocess_fn(x)
      with torch.no_grad():
            #print(i*batch_size)
            activations = ema_vae.encoder.forward(data_input)
            px_z, stats = ema_vae.decoder.forward(activations, get_latents=True)
            #recons = ema_vae.decoder.out_net.sample(px_z)
            batch_latent = []
            for i in range(num_latents):
                #test_latents[i].append(stats[i]['z'].cpu().numpy())
                batch_latent.append(stats[i]['z'].cpu().numpy().reshape(len(data_input),-1))
            latents.append(np.hstack(batch_latent))
    latents = np.concatenate(latents) 
    
    return latents



##### SET PATHS
base_path = '' #'/Projects/brain-diffuser/' #/home/lveronese
processed_dir = base_path + 'data/04_processed_data/subj_{}/'.format(sub)
extracted_dir = base_path + 'data/05_extracted_features/'



##### LOAD PRETRAINED MODEL
H = {'image_size': 64, 'image_channels': 3,'seed': 0, 'port': 29500, 'save_dir': './saved_models/test', 'data_root': './', 'desc': 'test', 'hparam_sets': 'imagenet64', 'restore_path': 'imagenet64-iter-1600000-model.th', 'restore_ema_path': 'vdvae/model/imagenet64-iter-1600000-model-ema.th', 'restore_log_path': 'imagenet64-iter-1600000-log.jsonl', 'restore_optimizer_path': 'imagenet64-iter-1600000-opt.th', 'dataset': 'imagenet64', 'ema_rate': 0.999, 'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5', 'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12', 'zdim': 16, 'width': 512, 'custom_width_str': '', 'bottleneck_multiple': 0.25, 'no_bias_above': 64, 'scale_encblock': False, 'test_eval': True, 'warmup_iters': 100, 'num_mixtures': 10, 'grad_clip': 220.0, 'skip_threshold': 380.0, 'lr': 0.00015, 'lr_prior': 0.00015, 'wd': 0.01, 'wd_prior': 0.0, 'num_epochs': 10000, 'n_batch': 4, 'adam_beta1': 0.9, 'adam_beta2': 0.9, 'temperature': 1.0, 'iters_per_ckpt': 25000, 'iters_per_print': 1000, 'iters_per_save': 10000, 'iters_per_images': 10000, 'epochs_per_eval': 1, 'epochs_per_probe': None, 'epochs_per_eval_save': 1, 'num_images_visualize': 8, 'num_variables_visualize': 6, 'num_temperatures_visualize': 3, 'mpi_size': 1, 'local_rank': 0, 'rank': 0, 'logdir': './saved_models/test/log'}
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
H = dotdict(H)

H, preprocess_fn = set_up_data(H)

print('Model is Loading...')
ema_vae = load_vaes(H)



##### DEFINE CLASS FOR DATASET ITERATION
class batch_generator_external_images(Dataset):
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.im = np.load(data_path).astype(np.uint8)
        print(self.im.shape)

    def __getitem__(self,idx):
        img = Image.fromarray(self.im[idx])
        #img.save(base_path + 'results/tests/' + f"image_{idx}_before_resize.png")
        img = T.functional.resize(img,(64,64))
        #img.save(base_path + 'results/tests/' + f"image_{idx}_after_resize.png")
        img = torch.tensor(np.array(img)).float()
        #img = img/255
        #img = img*2 - 1
        return img

    def __len__(self):
        return  len(self.im)



##### PREPARE DATA LOADING
print('Train stimuli of shape: ')
stims_train = batch_generator_external_images(processed_dir + 'stimuli_train.npy')
print('Test stimuli of shape: ')
stims_test = batch_generator_external_images(processed_dir + 'stimuli_test.npy')


##### EXTRACT LATENT VARIABLES USING VDVAE
train_loader = DataLoader(stims_train, batch_size, shuffle=False)
test_loader = DataLoader(stims_test, batch_size, shuffle=False)
num_latents = 31

print()
print("##############################")
print('Extracting the latents variables using the VDVAE for the train set...')
train_latents = extract_latents(train_loader, num_latents)
print('Resulting train latents of shape: ', train_latents.shape)

print()
print("##############################")
print('Extracting the latents variables using the VDVAE for the test set...')
test_latents = extract_latents(test_loader, num_latents)
print('Resulting test latents of shape: ', test_latents.shape)

print()
print("##############################")
save_dir = extracted_dir + 'subj_{}/'.format(sub)
save_filename = 'vdvae_latents.npz'
save_full_path = save_dir + save_filename
print('Saving at "' + save_full_path + '"...')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
np.savez(save_full_path, train_latents=train_latents, test_latents=test_latents)