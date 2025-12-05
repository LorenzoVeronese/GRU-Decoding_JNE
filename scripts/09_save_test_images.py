import numpy as np
import os
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)


##### SET PATHS
base_path = '' 
processed_dir = base_path + 'data/04_processed_data/subj_{}/'.format(sub)
gt_results_dir = base_path + 'results/ground_truth/subj_{}/'.format(sub)



##### SAVE
images = np.load(processed_dir + 'stimuli_test.npy')

if not os.path.exists(gt_results_dir):
    os.makedirs(gt_results_dir)

if not os.path.exists(gt_results_dir):
   os.makedirs(gt_results_dir)
for i in range(len(images)):
    im = Image.fromarray(images[i].astype(np.uint8))
    im.save(gt_results_dir + '{}.png'.format(i))


