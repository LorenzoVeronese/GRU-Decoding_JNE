import sys
import numpy as np
import sklearn.linear_model as skl
import pickle
import argparse
import os
from utils import metrics_vector_compute
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-alpha", "--alpha",help="Regularization of Ridge", default=100000)
args = parser.parse_args()
sub=int(args.sub)
alpha=int(args.alpha)



##### SET PATHS
base_path = ''
processed_dir = base_path + 'data/04_processed_data/'
extracted_dir = base_path + 'data/05_extracted_features/'
regression_dir = base_path + 'data/06_regression_weights/subj_{}/'.format(sub)
predicted_dir = base_path + 'data/07_predicted_features/subj_{}/'.format(sub)
results_metrics_dir = base_path + 'results/metrics/'




##### GET CLIP-TEXT EMBEDDINGS
print()
print("##############################")
print('Retrieving CLIP-text embeddings...')
train_clip = np.load(extracted_dir + 'subj_{}/cliptext_train.npy'.format(sub))
test_clip = np.load(extracted_dir + 'subj_{}/cliptext_test.npy'.format(sub))



##### GET BETAS
print()
print("##############################")
print('Retrieving betas...')
train_betas = np.load(processed_dir + 'subj_{}/betas_train.npy'.format(sub))
test_betas = np.load(processed_dir + 'subj_{}/betas_test.npy'.format(sub))



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
num_samples,num_embed,num_dim = train_clip.shape

print('Retrieving best alphas found with grid search...')
file_path = regression_dir + 'cliptext_best_alphas.npy'
if os.path.exists(file_path):
    best_alphas = np.load(file_path)
    print('Alphas that are going to be used:', best_alphas)
else:
    default_alpha = alpha
    print('File not found. Using default alpha: ', default_alpha)
    best_alphas = np.full(num_embed, default_alpha)


##### TRAIN REGRESSION
print()
print("##############################")
print('Training regression...')
num_voxels, num_train, num_test = train_betas.shape[1], len(train_betas), len(test_betas)

reg_w = np.zeros((num_embed,num_dim,num_voxels)).astype(np.float32)
reg_b = np.zeros((num_embed,num_dim)).astype(np.float32)
pred_clip = np.zeros_like(test_clip) 
pred_clip_nozscore = np.zeros_like(test_clip)
r2s = []
for i in range(num_embed):
    reg = skl.Ridge(alpha=best_alphas[i], max_iter=50000, fit_intercept=True) #default alpha: see in input arguments
    reg.fit(train_betas, train_clip[:,i])
    reg_w[i] = reg.coef_
    reg_b[i] = reg.intercept_
    
    pred_test_clips = reg.predict(test_betas)
    pred_clip_nozscore[:, i] = pred_test_clips
    std_norm_test_clips = (pred_test_clips - np.mean(pred_test_clips,axis=0)) / np.std(pred_test_clips,axis=0)
    pred_clip[:,i] = std_norm_test_clips * np.std(train_clip[:,i],axis=0) + np.mean(train_clip[:,i],axis=0)
    
    r2s.append(reg.score(test_betas,test_clip[:,i]))
    print(f'Model {i+1}', r2s[-1])

flattened_test_clip = test_clip.reshape(len(test_clip), -1)
print(flattened_test_clip.shape)
flattened_pred_clip_nozscore = pred_clip_nozscore.reshape(len(test_clip), -1)
print(flattened_pred_clip_nozscore.shape)
# Imported from scripts/utils
print(f'Mean R2: {np.mean(r2s)}')
metrics_vector_compute.compute_metrics(flattened_test_clip, flattened_pred_clip_nozscore, np.mean(r2s), results_metrics_dir,'CLIPtext', sub, alpha)
flattened_pred_clip = pred_clip.reshape(len(test_clip), -1) 
metrics_vector_compute.compute_metrics(flattened_test_clip, flattened_pred_clip, np.mean(r2s), results_metrics_dir,'CLIPtextZscoreInverted', sub, alpha)



###### SAVE
print()
print("##############################")
# Save predicted test latent variables
if not os.path.exists(predicted_dir):
    os.makedirs(predicted_dir)
predicted_filename = 'cliptext_pred.npy'
print(f'Saving the predicted CLIP-text embeddings ({predicted_filename})...')
np.save(predicted_dir + predicted_filename, pred_clip)


# Save regression weights
if not os.path.exists(regression_dir):
    os.makedirs(regression_dir)
datadict = {
    'weight' : reg.coef_,
    'bias' : reg.intercept_,
}
regression_filename = f'cliptext_regression_weights_alpha{alpha}.pkl'
print(f'Saving the regression weights ({regression_filename})...')
with open(regression_dir + regression_filename,"wb") as f:
    pickle.dump(datadict,f)
    
    
    
    
    
    
    
    
    
    
    
    
