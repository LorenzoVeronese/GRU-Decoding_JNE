import os
import sys
import numpy as np
import h5py
import scipy.io as spio
import nibabel as nib
import scipy as sp
from PIL import Image
from scipy.stats import pearsonr,binom,linregress
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
import pandas as pd
# Added
from scipy import stats
import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)



##### FUNCTIONS
def pairwise_corr_all(ground_truth, predictions):
    """
    The pairwise_corr_all function is performing a pairwise correlation analysis between ground truth features 
    and predicted features. Here's a breakdown of what it's doing:

    1) It computes the correlation coefficient matrix between ground truth and predictions.
    2) It extracts the diagonal of this matrix, which represents the correlation between each ground truth feature and 
        its corresponding predicted feature.
        np.corrcoef() computes correlations between all pairs of rows in this combined array. The resulting correlation matrix includes:
            Correlations of ground_truth with itself
            Correlations of predictions with itself
            Correlations between ground_truth and predictions
            Correlations between predictions and ground_truth
        so given 65xsomething in input, output is 330x330.
    3) It then compares each correlation in the matrix to the corresponding diagonal value. A "success" is when the 
        off-diagonal correlation is lower than the diagonal correlation.
    4) It calculates the average success rate across all comparisons.
    5) Finally, it computes a p-value using a binomial test to assess the statistical significance of the success rate.
    """
    # Get pairwise correlation
    r = np.corrcoef(ground_truth, predictions) #cosine_similarity(ground_truth, predictions)#
    r = r[:len(ground_truth), len(ground_truth):]  # upper-right matrix: rows=groundtruth, columns=predicitons
    
    # Congruent pairs are on diagonal
    congruents = np.diag(r)
    mean_congruents = np.mean(congruents)
    std_congruents = np.std(congruents)
    median_congruents = np.median(congruents)
    
    # for each column (predicition) we should count the number of rows (groundtruth) that the value is lower than the congruent (e.g. success).
    success = r < congruents
    success_cnt = np.sum(success, 0)
    
    # note: diagonal of 'success' is always zero so we can discard it. That's why we divide by len-1
    perf = np.mean(success_cnt) / (len(ground_truth)-1)
    p = 1 - binom.cdf(perf*len(ground_truth)*(len(ground_truth)-1), len(ground_truth)*(len(ground_truth)-1), 0.5)
    
    return perf, p, mean_congruents, std_congruents, median_congruents



import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
def visualize_corrcoef(ground_truth, predictions, net_name, layer):
    # Compute the correlation coefficient matrix
    r = np.corrcoef(ground_truth, predictions)
    r = r[:len(ground_truth), len(ground_truth):]
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create a heatmap
    im = ax.imshow(r, cmap='coolwarm', norm=Normalize(vmin=-1, vmax=1))
    
    # Add colorbar
    cbar = fig.colorbar(im)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15)
    
    # Set title and labels
    ax.set_title('Correlation Coefficient Matrix Heatmap', fontsize=16)
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    
    # Add lines to separate ground truth and predictions
    ax.axhline(y=len(ground_truth) - 0.5, color='k', linestyle='--')
    ax.axvline(x=len(ground_truth) - 0.5, color='k', linestyle='--')
    
    # Add text annotations
    ax.text(-0.05 * len(r), len(ground_truth) / 2, 'Ground Truth', 
            rotation=90, va='center', ha='center', fontsize=12)
    ax.text(-0.05 * len(r), len(ground_truth) + len(predictions) / 2, 'Predictions', 
            rotation=90, va='center', ha='center', fontsize=12)
    ax.text(len(ground_truth) / 2, -0.05 * len(r), 'Ground Truth', 
            va='center', ha='center', fontsize=12)
    ax.text(len(ground_truth) + len(predictions) / 2, -0.05 * len(r), 'Predictions', 
            va='center', ha='center', fontsize=12)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(results_metrics_dir + f'subj_{sub}/paircorr_heatmap_{net_name}-{str(layer)}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the plot to free up memory



##### SET PATHS
base_path = '' 
processed_dir = base_path + 'data/04_processed_data/'
extracted_dir = base_path + 'data/05_extracted_features/'
regression_dir = base_path + 'data/06_regression_weights/'
predicted_dir = base_path + 'data/07_predicted_features/'
results_vae_dir = base_path + 'results/vdvae/'
results_versatile_dir = base_path + 'results/versatile_diffusion/'
results_gt_dir = base_path + 'results/ground_truth/'
results_metrics_dir = base_path + 'results/metrics/'
eval_features_dir = base_path + 'data/08_eval_features/'

feats_dir = eval_features_dir + 'subj_{}/'.format(sub)



##### SET FEATURES TO COMPARE AND PARAMETERS
net_list = [
    ('inceptionv3','avgpool'),
    ('clip','final'),
    ('alexnet',2),
    ('alexnet',5),
    ('efficientnet','avgpool'),
    ('swav','avgpool')
]

test_images = np.load(processed_dir + 'subj_{}/stimuli_test.npy'.format(sub))
num_test = len(test_images)#65
distance_fn = sp.spatial.distance.correlation



##### CREATE A DF TO PUT RESULTS IN
columns = ['Name', 'Sub-name', 'Mean', 'Standard deviation', 'Median']
df_results = pd.DataFrame(columns=columns)



print()
print("##############################")
print('High-level metrics:')
pairwise_corrs = []
for (net_name,layer) in net_list:
    file_name = feats_dir + 'ground_truth/{}_{}.npy'.format(net_name,layer)
    gt_feat = np.load(file_name)
    
    file_name = feats_dir + '{}_{}.npy'.format(net_name,layer)
    eval_feat = np.load(file_name)
    
    gt_feat = gt_feat.reshape((len(gt_feat),-1))
    eval_feat = eval_feat.reshape((len(eval_feat),-1))
    #Shapes:
    # inceptionv3 avgpool: (., 2048)
    # clip final: (., 768)
    # alexnet 2: (., 139968)
    # alexnet 5: (., 43264)
    # efficientnet avgpool: (., 1280)
    # swav avgpool: (., 2048)
    
    print(net_name, layer)
    if net_name in ['efficientnet','swav']:
        distances = np.array([distance_fn(gt_feat[i],eval_feat[i]) for i in range(num_test)])
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        median_distance = np.median(distances)
        
        df_results.loc[len(df_results)] = [net_name+str(layer), None, mean_distance, std_distance, median_distance]
        print('    distance: Mean={}, Std Dev={}, Median={}'.format(mean_distance, std_distance, median_distance))
        print()

    else:
        perf, p, mean_congruents, std_congruents, median_congruents = pairwise_corr_all(gt_feat, eval_feat)
        pairwise_corrs.append(perf)
        visualize_corrcoef(gt_feat, eval_feat, net_name, layer)
    
        df_results.loc[len(df_results)] = [net_name+str(layer), 'two-way', pairwise_corrs[-1], None, None]
        df_results.loc[len(df_results)] = [net_name, 'correlation', mean_congruents, std_congruents, median_congruents]
        print('    pairwise corr: ', pairwise_corrs[-1])
        
        print('    correlations: Mean={}, Std Dev={}, Median={}'.format(mean_congruents, std_congruents, median_congruents))
        print()

   
print()
print("##############################")
print('Low-level metrics:')     
ssim_list = []
pixcorr_list = []
for i in range(num_test):
    gen_image = Image.open(results_versatile_dir + 'subj_{}/{}.png'.format(sub, i)).resize((425,425))
    gt_image = Image.open(results_gt_dir + 'subj_{}/{}.png'.format(sub, i)).resize((425, 425))
    gen_image = np.array(gen_image)/255.0
    gt_image = np.array(gt_image)/255.0
    
    
    
    pixcorr_res = np.corrcoef(gt_image.reshape(1,-1), gen_image.reshape(1,-1))[0,1]
    pixcorr_list.append(pixcorr_res)
    
    gen_image = rgb2gray(gen_image)
    gt_image = rgb2gray(gt_image)
    ssim_res = ssim(gen_image, gt_image, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
    ssim_list.append(ssim_res)

# Convert lists to numpy arrays
pixcorr_array = np.array(pixcorr_list)
ssim_array = np.array(ssim_list)

# Calculate means and standard deviations
pixcorr_mean = np.mean(pixcorr_array)
pixcorr_std = np.std(pixcorr_array)
pixcorr_median = np.median(pixcorr_array)
ssim_mean = np.mean(ssim_array)
ssim_std = np.std(ssim_array)
ssim_median = np.median(ssim_array)

df_results.loc[len(df_results)] = ['pix_corr', None, pixcorr_mean, pixcorr_std, pixcorr_median]
df_results.loc[len(df_results)] = ['ssim', None, ssim_mean, ssim_std, ssim_median]
print('PixCorr: Mean={}, Std Dev={}, Median={}'.format(pixcorr_mean, pixcorr_std, pixcorr_median))
print()
print('SSIM: Mean={}, Std Dev={}, Median={}'.format(ssim_mean, ssim_std, ssim_median))
print()
#print(df_results)
if not os.path.exists(results_metrics_dir + 'subj_{}'.format(sub)):
    os.makedirs(results_metrics_dir + 'subj_{}'.format(sub))
df_results.to_csv(results_metrics_dir + 'subj_{}/metrics_full_reconstruction-sub{}.csv'.format(sub, sub), index=False)




