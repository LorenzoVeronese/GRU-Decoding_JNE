import os
import pandas as pd
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Don't wrap to multiple lines
pd.set_option('display.max_colwidth', None)  # Show full content of each column

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-alpha_vdvae", "--alpha_vdvae",help="Alpha for VDVAE regressor",default=50000)
parser.add_argument("-alpha_cliptext", "--alpha_cliptext",help="Alpha for CLIP-text regressor",default=100000)
parser.add_argument("-alpha_clipvision", "--alpha_clipvision",help="Alpha for CLIP-vision regressor",default=60000)
args = parser.parse_args()
sub=int(args.sub)
alpha_vdvae=int(args.alpha_vdvae)
alpha_cliptext=int(args.alpha_cliptext)
alpha_clipvision=int(args.alpha_clipvision)
allowed_subjs = [i for i in range(1, 27) if i != 10]
assert sub in allowed_subjs

##### SET PATHS
base_path = ''
results_metrics_dir = base_path + 'results/metrics/'



csv_paths = [
    results_metrics_dir + 'subj_{}/metrics_CLIPtext-sub{}-alpha{}.csv'.format(sub, sub, alpha_cliptext), # CLIP-text regression
    results_metrics_dir + 'subj_{}/metrics_CLIPtextZscoreInverted-sub{}-alpha{}.csv'.format(sub, sub, alpha_cliptext), # CLIP-text regression (z-score inverted)
    results_metrics_dir + 'subj_{}/metrics_CLIPvision-sub{}-alpha{}.csv'.format(sub, sub, alpha_clipvision), # CLIP-vision regression
    results_metrics_dir + 'subj_{}/metrics_CLIPvisionZscoreInverted-sub{}-alpha{}.csv'.format(sub, sub, alpha_clipvision), # CLIP-vision regression (z-score inverted
    results_metrics_dir + 'subj_{}/metrics_full_reconstruction-sub{}.csv'.format(sub, sub) #full reconstruction
]
for csv_path in csv_paths:  
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Add the DataFrame to our list
    print('############################################################')
    print('From file {}'.format(os.path.basename(csv_path)))
    print(df)
    print()
