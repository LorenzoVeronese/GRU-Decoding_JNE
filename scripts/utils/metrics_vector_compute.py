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

##### FUNCTIONS
import matplotlib.pyplot as plt
def plot_aggregate_samples(aggregate_samples, output_file):
    plt.figure(figsize=(12, 6))

    # Create a histogram
    n, bins, patches = plt.hist(aggregate_samples, bins=100, edgecolor='black')

    # Customize the plot
    plt.title('Distribution of Metric Values across Latent Dimensions', fontsize=16)
    plt.xlabel('Metric Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    # Add vertical lines for mean and median
    mean_value = np.mean(aggregate_samples)
    plt.axvline(x=mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.2f}%')

    # Add legend
    plt.legend()

    # Improve layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()



def plot_aggregate_scatter(data, filename, y_max=None):
    """
    Saves a scatter plot of a 1D NumPy array to a file.

    Parameters:
    data (numpy.ndarray): 1D array of data points to plot.
    filename (str): The filename to save the plot. Default is 'scatter_plot.png'.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 1:
        raise ValueError("Input data must be a 1D NumPy array")

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    if not y_max==None: plt.ylim(0, y_max)
    plt.scatter(range(len(data)), data, c='blue', alpha=0.5, s = 2)
    plt.title('Scatter Plot of 1D NumPy Array')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)

    # Save the plot as an image file
    plt.savefig(filename)
    plt.close()



def compute_metrics(gt, pred, r2, results_metrics_dir, technique, sub, alpha):
    """
    gt: ground truth
    pred: predictions
    r2: r-squared, just for saving it in the .csv
    results_metrics_dir: (e.g base_path + 'results/metrics/') path up to folder of all
    technique: (e.g. 'latents', 'CLIPtxt')
    sub: (e.g. 1)
    alpha: (e.g. 50000) regularization level used for Ridge
    """
    """
    They are is n_samples x n_latents matrices
    """
    ##### CREATE A DF TO PUT RESULTS IN
    columns = ['Category', 'Metric', 'Mean', 'Standard deviation']
    df_results = pd.DataFrame(columns=columns)
    
    df_results.loc[len(df_results)] = ['R2', 'R2', r2, None]
    
    ##### PERCENTAGE ERRORS
    epsilon = 1e-10 #small value to avoid division by zero
    '''#PMeanSE
    pmeanse_samples = np.mean((((gt - pred)/gt)**2)*100, axis=0)
    pmeanse = np.mean(pmeanse_samples)
    pmeanse_std = np.std(pmeanse_samples)
    
    #PMedianSE
    pmedianse_samples = np.median((((gt - pred)/gt)**2)*100, axis=0)
    pmedianse = np.mean(pmedianse_samples)
    pmedianse_std = np.std(pmedianse_samples)'''
    perc_thresh = 100
    #PMeanAE
    pmeanae_samples = np.mean((np.abs((gt - pred)/(gt+epsilon)))*100, axis=0)
    count_pmeanae = np.sum(pmeanae_samples < perc_thresh)/len(pmeanae_samples)
    pmeanae = np.mean(pmeanae_samples)
    pmeanae_std = np.std(pmeanae_samples)
    
    #PMedianAE
    pmedianae_samples = np.median((np.abs((gt - pred)/(gt+epsilon)))*100, axis=0)
    count_pmedianae = np.sum(pmedianae_samples < perc_thresh)/len(pmedianae_samples)
    pmedianae = np.mean(pmedianae_samples)
    pmedianae_std = np.std(pmedianae_samples)
    
    print('Percentage errors:')
    #print("    Test PMeanSE:", pmeanse, "- Std dev:", pmeanse_std)
    #print("    Test PMedianSE:", pmedianse, "- Std dev:", pmedianse_std)
    print("    Test PMeanAE:", pmeanae, "- Std dev:", pmeanae_std)
    print("    Test PMeanAE count <"+str(perc_thresh)+":", count_pmeanae)
    print("    Test PMedianAE:", pmedianae, "- Std dev:", pmedianae_std)
    print("    Test PMedianAE count <"+str(perc_thresh)+":", count_pmedianae)
    print(f'    Example of PMeanAE per feature: {pmeanae_samples[0:5]}')
    df_results.loc[len(df_results)] = ['percentage errors', 'PMeanAE', pmeanae, pmeanae_std]
    df_results.loc[len(df_results)] = ['percentage errors', f'count PMeanAE <{perc_thresh}', count_pmeanae, None]
    df_results.loc[len(df_results)] = ['percentage errors', 'PMedianAE', pmedianae, pmedianae_std]   
    df_results.loc[len(df_results)] = ['percentage errors', f'count PMedianAE <{perc_thresh}', count_pmedianae, None]
    if not os.path.exists(results_metrics_dir+ f'subj_{sub}/'):
        os.makedirs(results_metrics_dir+ f'subj_{sub}/')
    plot_aggregate_samples(pmeanae_samples, results_metrics_dir + 'subj_{}/metrics_{}_percentage_pmeanae-sub{}-alpha{}.png'.format(sub, technique, sub, alpha))
    plot_aggregate_samples(pmedianae_samples, results_metrics_dir + 'subj_{}/metrics_{}_percentage_pmedianae-sub{}-alpha{}.png'.format(sub, technique, sub, alpha))
    plot_aggregate_scatter(pmeanae_samples, results_metrics_dir + 'subj_{}/metrics_{}_percentage_scatter_pmeanae-sub{}-alpha{}.png'.format(sub, technique, sub, alpha))
    plot_aggregate_scatter(pmedianae_samples, results_metrics_dir + 'subj_{}/metrics_{}_percentage_scatter_pmedianae-sub{}-alpha{}.png'.format(sub, technique, sub, alpha))
    plot_aggregate_scatter(pmeanae_samples, results_metrics_dir + 'subj_{}/metrics_{}_percentage_scatter-CUT_pmeanae-sub{}-alpha{}.png'.format(sub, technique, sub, alpha), 5000)
    plot_aggregate_scatter(pmedianae_samples, results_metrics_dir + 'subj_{}/metrics_{}_percentage_scatter-CUT_pmedianae-sub{}-alpha{}.png'.format(sub, technique, sub, alpha), 200)
    
    

    # WITHOUT NORMALIZATION
    # MSE
    mse_samples = np.mean((pred - gt)**2, axis=0)
    mse = np.mean(mse_samples)
    mse_std = np.std(mse_samples)
    
    # RMSE
    rmse_samples = np.sqrt(np.mean((pred - gt)**2, axis=0))
    rmse = np.mean(rmse_samples)
    rmse_std = np.std(rmse_samples)
    
    # MedianSE
    medianse_samples = np.median((pred - gt)**2, axis=0)
    medianse = np.mean(medianse_samples)
    medianse_std = np.std(medianse_samples)
    
    # RMedianSE
    rmedianse_samples = np.sqrt(np.median((pred - gt)**2, axis=0))
    rmedianse = np.mean(rmedianse_samples)
    rmedianse_std = np.std(rmedianse_samples)
    
    # MeanAE
    meanae_samples = np.mean(np.abs(pred - gt), axis=0)
    meanae = np.mean(meanae_samples)
    meanae_std = np.std(meanae_samples)
    
    # MedianAE
    medianae_samples = np.median(np.abs(pred - gt), axis=0)
    medianae = np.mean(medianae_samples)
    medianae_std = np.std(medianae_samples)
    
    print('Before normalization (feature-wise):')
    print("    Test MeanSE:", mse, "- Std dev:", mse_std)
    print("    Test RMeanSE:", rmse, "- Std dev:", rmse_std)
    print("    Test MedianSE:", medianse, "- Std dev:", medianse_std)
    print("    Test RMedianSE:", rmedianse, "- Std dev:", rmedianse_std)
    print("    Test MeanAE:", meanae, "- Std dev:", meanae_std)
    print("    Test MedianAE:", medianae, "- Std dev:", medianae_std)
    print(f'    Example of MSE per feature: {mse_samples[0:5]}')
    df_results.loc[len(df_results)] = ['no normalization', 'MeanSE', mse, mse_std]
    df_results.loc[len(df_results)] = ['no normalization', 'RMeanSE', rmse, rmse_std]
    df_results.loc[len(df_results)] = ['no normalization', 'MedianSE', medianse, medianse_std]
    df_results.loc[len(df_results)] = ['no normalization', 'RMedianSE', rmedianse, rmedianse_std]
    df_results.loc[len(df_results)] = ['no normalization', 'MeanAE', meanae, meanae_std]
    df_results.loc[len(df_results)] = ['no normalization', 'MedianAE', medianae, medianae_std]

    
    
    # WITH NORMALIZATION
    # Stack matrices
    n_rows = len(gt)
    stacked = np.vstack((gt, pred))
    
    # Normalize columns (z-score)
    for col in range(0, stacked.shape[1]):
        stacked[:, col] = (stacked[:, col]-np.mean(stacked[:, col]))/np.std(stacked[:, col])
        
    # Re-split the matrices
    gt = stacked[:n_rows, :]
    pred = stacked[n_rows:, ]
    
    
    # MSE
    mse_samples = np.mean((pred - gt)**2, axis=0)
    mse = np.mean(mse_samples)
    mse_std = np.std(mse_samples)
    
    # RMSE
    rmse_samples = np.sqrt(np.mean((pred - gt)**2, axis=0))
    rmse = np.mean(rmse_samples)
    rmse_std = np.std(rmse_samples)
    
    # MedianSE
    medianse_samples = np.median((pred - gt)**2, axis=0)
    medianse = np.mean(medianse_samples)
    medianse_std = np.std(medianse_samples)
    
    # RMedianSE
    rmedianse_samples = np.sqrt(np.median((pred - gt)**2, axis=0))
    rmedianse = np.mean(rmedianse_samples)
    rmedianse_std = np.std(rmedianse_samples)
    
    # MeanAE
    meanae_samples = np.mean(np.abs(pred - gt), axis=0)
    meanae = np.mean(meanae_samples)
    meanae_std = np.std(meanae_samples)
    
    # MedianAE
    medianae_samples = np.median(np.abs(pred - gt), axis=0)
    medianae = np.mean(medianae_samples)
    medianae_std = np.std(medianae_samples)

    
    print('After Z-score normalization (feature-wise):')
    print("    Test MeanSE:", mse, "- Std dev:", mse_std)
    print("    Test RMeanSE:", rmse, "- Std dev:", rmse_std)
    print("    Test MedianSE:", medianse, "- Std dev:", medianse_std)
    print("    Test RMedianSE:", rmedianse, "- Std dev:", rmedianse_std)
    print("    Test MeanAE:", meanae, "- Std dev:", meanae_std)
    print("    Test MedianAE:", medianae, "- Std dev:", medianae_std)
    print(f'    Example of MSE per feature: {mse_samples[0:5]}')
    df_results.loc[len(df_results)] = ['z-score (feature-wise)', 'MeanSE', mse, mse_std]
    df_results.loc[len(df_results)] = ['z-score (feature-wise)', 'RMeanSE', rmse, rmse_std]
    df_results.loc[len(df_results)] = ['z-score (feature-wise)', 'MedianSE', medianse, medianse_std]
    df_results.loc[len(df_results)] = ['z-score (feature-wise)', 'RMedianSE', rmedianse, rmedianse_std]
    df_results.loc[len(df_results)] = ['z-score (feature-wise)', 'MeanAE', meanae, meanae_std]
    df_results.loc[len(df_results)] = ['z-score (feature-wise)', 'MedianAE', medianae, medianae_std]
    
    print('Saving the csv with all metrics...')
    df_results.to_csv(results_metrics_dir + 'subj_{}/metrics_{}-sub{}-alpha{}.csv'.format(sub, technique, sub, alpha), index=False)
    
    
    
    
    
    
    
