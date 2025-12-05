import sys
sys.path.append('versatile_diffusion')
import os
import os.path as osp
import PIL
from PIL import Image
from pathlib import Path
import numpy as np
import numpy.random as npr
import torch
import torchvision.transforms as tvtrans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD
from lib.experiments.sd_default import color_adjust, auto_merge_imlist
from torch.utils.data import DataLoader, Dataset
from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
import matplotlib.pyplot as plt
from skimage.transform import resize, downscale_local_mean
import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub", help="Subject Number", default=1)
parser.add_argument("-use_other_model_as_first_step", "--use_other_model_as_first_step", help="0: use vdvae as first step", default=0)
parser.add_argument("-diff_str", "--diff_str", help="Diffusion Strength",default=0.75) #original: 0.75
parser.add_argument("-mix_str", "--mix_str", help="Mixing Strength",default=0.4) #original: 0.4
args = parser.parse_args()
sub=int(args.sub)
use_other_model_as_first_step=int(args.use_other_model_as_first_step)
strength = float(args.diff_str)
mixing = float(args.mix_str)




##### FUNCTIONS
def regularize_image(x):
        BICUBIC = PIL.Image.Resampling.BICUBIC
        if isinstance(x, str):
            x = Image.open(x).resize([512, 512], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, PIL.Image.Image):
            x = x.resize([512, 512], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, np.ndarray):
            x = PIL.Image.fromarray(x).resize([512, 512], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, torch.Tensor):
            pass
        else:
            assert False, 'Unknown image type'

        assert (x.shape[1]==512) & (x.shape[2]==512), \
            'Wrong image size'
        return x



##### SET PATHS
base_path = '' #'/Projects/brain-diffuser/'
processed_dir = base_path + 'data/04_processed_data/'
extracted_dir = base_path + 'data/05_extracted_features/'
regression_dir = base_path + 'data/06_regression_weights/'
predicted_dir = base_path + 'data/07_predicted_features/'
results_vae_dir = base_path + 'results/vdvae/'
results_versatile_dir = base_path + 'results/versatile_diffusion/'



##### LOAD PRETRAINED MODEL
cfgm_name = 'vd_noema' #means that no EMA (exponential moving average is used)
sampler = DDIMSampler_VD
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)    

# Might require editing the GPU assignments due to Memory issues
net.clip.cuda(0)
net.autokl.cuda(0)

#net.model.cuda(1)
sampler = sampler(net)
#sampler.model.model.cuda(1)
#sampler.model.cuda(1)
batch_size = 1

pred_text = np.load(predicted_dir + 'subj_{}/cliptext_pred.npy'.format(sub))
pred_text = torch.tensor(pred_text).half().cuda(0)

pred_vision = np.load(predicted_dir + 'subj_{}/clipvision_pred.npy'.format(sub))
pred_vision = torch.tensor(pred_vision).half().cuda(0)


n_samples = 1
ddim_steps = 50
ddim_eta = 0
scale = 7.5
xtype = 'image'
ctype = 'prompt'
net.autokl.half()

# Delete old images or create folder for new ones
if not os.path.exists(results_versatile_dir + 'subj_{}/'.format(sub)):
    os.makedirs(results_versatile_dir + 'subj_{}/'.format(sub))
else:
    print("Deleting old images at " + results_versatile_dir + 'subj_{}/'.format(sub))
    for file_name in os.listdir(results_versatile_dir + 'subj_{}/'.format(sub)):
        file_path = results_versatile_dir + 'subj_{}/'.format(sub) + file_name
        os.remove(file_path)

torch.manual_seed(0)
for im_id in range(len(pred_vision)):

    zim = Image.open(results_vae_dir + 'subj_{}/{}.png'.format(sub,im_id))
    
    '''##### TRIALS WITH NOISE ON IMAGE. Uncomment to test different levels of noise applied to the first-stage reconstruction (see paper)
    #partial noise:
    #0.03, 0.06, 0.12 in scale 1
    #8, 16, 31 in scale 255
    zim = np.array(zim)
    max_val_noise = 64

    noise = np.random.normal(0, 1, zim.shape) * max_val_noise
    zim = zim + noise
    zim = np.clip(zim, 0, 255)
    zim = Image.fromarray(zim.astype('uint8'))
    
    # Save image
    noise_path = base_path + f'results/noise{max_val_noise}/' + 'subj_{}/'.format(sub)
    if not os.path.exists(noise_path):
        os.makedirs(noise_path)
    zim.save(noise_path + '{}.png'.format(im_id))'''

    ################################
   
    zim = regularize_image(zim) #works both if give PIL.Image and if np.array
    zin = zim*2 - 1

    """##### TRIALS WITH PURE NOISE. Uncomment to test pure noise instead of the first-stage reconstruction (see paper)
    max_val_noise = 1
    noise = np.random.normal(0, 1, (512, 512, 3)) * max_val_noise
    noise = torch.from_numpy(noise).float().permute(2, 0, 1)
    noise = (noise + 1) / 2  # Scale from [-1, 1] to [0, 1]

    # Convert to numpy array and scale to [0, 255]
    noise_np = (noise.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Create and save the image
    noise_img = Image.fromarray(noise_np)
    noise_path = os.path.join(base_path, f'results/noise256', f'subj_{sub}')
    os.makedirs(noise_path, exist_ok=True)
    noise_img.save(os.path.join(noise_path, f'{im_id}.png'))

    # If you want to use the noise as input
    zin = noise"""
    '''#pure noise (this doesn't save the images):
    max_val_noise = 1

    noise = np.random.normal(0, 1, zim.shape) * max_val_noise
    noise = tvtrans.ToTensor()(noise).float()
    noise = noise.permute(1, 2, 0)
    
    zin = noise
    
    # Save image
    noise_np = (zin.numpy() * 255).astype(np.uint8)
    # If the image is grayscale (1 channel), convert to RGB
    if noise_np.shape[-1] == 1:
        noise_np = np.repeat(noise_np, 3, axis=-1)
    # Create and save the image
    noise_img = Image.fromarray(noise_np)
    noise_path = base_path + f'results/noise{max_val_noise}' + 'subj_{}/'.format(sub)
    if not os.path.exists(noise_path):
        os.makedirs(noise_path)
    noise_img.save(noise_path + '{}.png'.format(im_id))'''
    
    #############################
    
    
    zin = zin.unsqueeze(0).cuda(0).half()

    init_latent = net.autokl_encode(zin)
    
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
    #strength=0.75
    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    device = 'cuda:0'
    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(device))
    #z_enc,_ = sampler.encode(init_latent.cuda(1).half(), c.cuda(1).half(), torch.tensor([t_enc]).to(sampler.model.model.diffusion_model.device))

    dummy = ''
    utx = net.clip_encode_text(dummy)
    utx = utx.cuda(0).half()
    
    dummy = torch.zeros((1,3,224,224)).cuda(0)
    uim = net.clip_encode_vision(dummy)
    uim = uim.cuda(0).half()
    
    z_enc = z_enc.cuda(0)

    h, w = 512,512
    shape = [n_samples, 4, h//8, w//8]

    cim = pred_vision[im_id].unsqueeze(0)
    ctx = pred_text[im_id].unsqueeze(0)
    
    #c[:,0] = u[:,0]
    #z_enc = z_enc.cuda(1).half()
    
    sampler.model.model.diffusion_model.device='cuda:0'
    sampler.model.model.diffusion_model.half().cuda(0)
    #mixing = 0.4
    
    z = sampler.decode_dc(
        x_latent=z_enc,
        first_conditioning=[uim, cim],
        second_conditioning=[utx, ctx],
        t_start=t_enc,
        unconditional_guidance_scale=scale,
        xtype='image', 
        first_ctype='vision',
        second_ctype='prompt',
        mixed_ratio=(1-mixing), )
    
    z = z.cuda(0).half()
    x = net.autokl_decode(z)
    color_adj='None'
    #color_adj_to = cin[0]
    color_adj_flag = (color_adj!='none') and (color_adj!='None') and (color_adj is not None)
    color_adj_simple = (color_adj=='Simple') or color_adj=='simple'
    color_adj_keep_ratio = 0.5
    
    if color_adj_flag and (ctype=='vision'):
        x_adj = []
        for xi in x:
            color_adj_f = color_adjust(ref_from=(xi+1)/2, ref_to=color_adj_to)
            xi_adj = color_adj_f((xi+1)/2, keep=color_adj_keep_ratio, simple=color_adj_simple)
            x_adj.append(xi_adj)
        x = x_adj
    else:
        x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)
        x = [tvtrans.ToPILImage()(xi) for xi in x]
    

    x[0].save(results_versatile_dir + 'subj_{}/{}.png'.format(sub, im_id))
      

