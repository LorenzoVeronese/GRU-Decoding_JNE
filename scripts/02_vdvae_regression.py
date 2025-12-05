import sys
import numpy as np
import sklearn.linear_model as skl
import argparse
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
import pickle
import pandas as pd
from utils import metrics_vector_compute


parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-alpha", "--alpha",help="Regularization of Ridge", default=50000)
parser.add_argument("-n_latents", "--n_latents", help="Number of latent variables", default=31)
args = parser.parse_args()
sub=int(args.sub)
alpha = int(args.alpha)
n_latents = int(args.n_latents)
assert n_latents in range(1, 32)



##### SET PATHS
base_path = '' #'/Projects/brain-diffuser/' #/home/lveronese
processed_dir = base_path + 'data/04_processed_data/subj_{}/'.format(sub)
extracted_dir = base_path + 'data/05_extracted_features/subj_{}/'.format(sub)
regression_dir = base_path + 'data/06_regression_weights/subj_{}/'.format(sub)
predicted_dir = base_path + 'data/07_predicted_features/subj_{}/'.format(sub)
results_metrics_dir = base_path + 'results/metrics/'



##### GET LATENT VARIABLES
print()
print("##############################")
print('Retrieving latent variables...')
latents_filename = 'vdvae_latents.npz'
latents = np.load(extracted_dir + latents_filename)
latents_hierarchy = [2**4,2**4,2**8,2**8,2**8,2**8,2**10,2**10,2**10,2**10,2**10,2**10,2**10,2**10,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**14]
cut_index = 0
for i in range(0, n_latents):
    cut_index += latents_hierarchy[i]
print(f'Latent variables cutted to dimension: {cut_index}')
train_latents = latents['train_latents'][:,:cut_index]
test_latents = latents['test_latents'][:,:cut_index]
print("Shape: ", train_latents.shape)



##### GET BETAS
print()
print("##############################")
print('Retrieving betas...')
train_betas = np.load(processed_dir + 'betas_train.npy')
test_betas = np.load(processed_dir + 'betas_test.npy')
print("Shape: ", train_betas.shape)



##### PREPROCESS (z-score)
print()
print("##############################")
print('Z-score normalization...')
#train_fmri = train_fmri/300
#test_fmri = test_fmri/300
epsilon = 1e-10
norm_mean_train = np.mean(train_betas, axis=0)
norm_std_train = np.std(train_betas, axis=0, ddof=1)

train_betas = (train_betas - norm_mean_train) / (norm_std_train+epsilon)
test_betas = (test_betas - norm_mean_train) / (norm_std_train+epsilon)

print("    Mean and standard deviation of training betas: ", np.mean(train_betas), np.std(train_betas))
print("    Mean and standard deviation of test betas: ", np.mean(test_betas), np.std(test_betas))
print("    Max and min of training betas: ", np.max(train_betas),np.min(train_betas))
print("    Max and min of test betas: ", np.max(test_betas),np.min(test_betas))



#### GET BEST ALPHAS OBTAINED BY GRID SEARCH
print()
print("##############################")
print('Retrieving best alpha found with grid search...')
file_path = regression_dir + 'vdvae_best_alpha.npy'
if os.path.exists(file_path):
    best_alpha = np.load(file_path)
    print('Alphas that are going to be used:', best_alpha)
else:
    default_alpha = alpha
    print('File not found. Using default alpha: ', default_alpha)
    best_alpha = default_alpha



##### TRAINING RIDGE REGRESSION
print()
print("##############################")
print(f'Training Ridge regression with alpha={alpha}...')
num_voxels, num_train, num_test = train_betas.shape[1], len(train_betas), len(test_betas)

# Train
reg = skl.Ridge(alpha=best_alpha, max_iter=10000, fit_intercept=True) #default alpha: see in input arguments
reg.fit(train_betas, train_latents) 


# NORMALIZE (z-score inverted) PREDICTED TEST LATENT VARIABLES TO TEST SCALE (can skip this)
pred_test_latents = reg.predict(test_betas)
print(f'Shape of test predictions: {pred_test_latents.shape}.')
std_norm_test_latents = (pred_test_latents - np.mean(pred_test_latents,axis=0)) / np.std(pred_test_latents,axis=0)
pred_latents = std_norm_test_latents * np.std(train_latents,axis=0) + np.mean(train_latents,axis=0)



##### GET TEST LATENT VARIABLES
print()
print("##############################")
print('Predicting test latent variables...')

r2 = reg.score(test_betas, test_latents)
print("    Test R^2: ", r2)
#imported from utils/metrics_compute_vector
if not os.path.exists(results_metrics_dir + 'subj_{}'.format(sub)):
    os.makedirs(results_metrics_dir + 'subj_{}'.format(sub))
metrics_vector_compute.compute_metrics(test_latents, pred_test_latents, r2, results_metrics_dir,'VDVAElatents', sub, alpha)
metrics_vector_compute.compute_metrics(test_latents, pred_latents, r2, results_metrics_dir,'VDVAElatentsZscoreInverted', sub, alpha)



###### SAVE
print()
print("##############################")
# Save predicted test latent variables
if not os.path.exists(predicted_dir):
    os.makedirs(predicted_dir)
predicted_filename = 'vdvae_pred.npy'
print(f'Saving the predicted test latent variables ({predicted_filename})...')
np.save(predicted_dir + predicted_filename, pred_latents)


# Save regression weights
if not os.path.exists(regression_dir):
    os.makedirs(regression_dir)
datadict = {
    'weight' : reg.coef_,
    'bias' : reg.intercept_,
}
regression_filename = 'vdvae_regression_weights.pkl'
print(f'Saving the regression weights ({regression_filename})...')
with open(regression_dir + regression_filename,"wb") as f:
    pickle.dump(datadict,f)
    
    
    
    


    
    
    
