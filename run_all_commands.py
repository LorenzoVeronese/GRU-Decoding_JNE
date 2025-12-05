import subprocess
from datetime import datetime
import os
import shutil
import glob


##### PARAMETERS
sub = 1 #[1, 2, 5, 7]
betas_type = '' #type of betas used: '' is the only current option
chosen_roi = 'nsdgeneral' #chosen ROI: nsdgeneral
n_latents = 15 # Number of latent variables of the VDVAE to be predicted. We found that for GRU-model, 15 is better then 31
alpha_vdvae = 50000 # Regularization factor for Ridge of VDVAE (for the case in which GRU-model is not used)
alpha_cliptext = 100000 # Regularization factor for Ridge of CLIP-text
alpha_clipvision = 60000 # Regularization factor for Ridge of CLIP-vision
use_other_model_as_first_step = 0 # 0: ridge on vdvae (as in brain-diffuser by Furkan Ozcelik and Rufin VanRullen), 2: GRU on vdvae
use_other_model_for_clip = 0 # 0: Ridge, 1: NN. Our results demonstrated that Ridge works better for CLIP predictions
normalize_pred_vdvae = 1 # Normalize predicted VDVAE embeddings of the test set. NOTE: for now it is implemented only for NN: elsewhere it's always 1
normalize_pred_clip = 1 # Normalize predicted CLIP embeddings of the test set. NOTE: for now it is implemented only for NN: elsewhere it's always 1



##### PATHS
base_path = '' # If the folder from which you are running the code is the same as the root of the project, just keep this empty
processed_dir = base_path + 'data/04_processed_data/'
extracted_dir = base_path + 'data/05_extracted_features/'
regression_dir = base_path + 'data/06_regression_weights/'
predicted_dir = base_path + 'data/07_predicted_features/'
results_metrics_dir = base_path + 'results/metrics/'
if use_other_model_as_first_step == 0 or use_other_model_as_first_step == 2: #VDVAE
    results_vae_dir = base_path + 'results/vdvae/'
results_versatile_dir = base_path + 'results/versatile_diffusion/'
results_metrics_dir = base_path + 'results/metrics/'
eval_features_dir = base_path + 'data/08_eval_features/'
results_gt_dir = base_path + 'results/ground_truth/'
saved_dir = base_path + 'saved_data/'



def run_command(command):
    """Run a shell command."""
    try:
        print(f"Running: {command}")
        subprocess.run(command, shell=True, check=True)
        print(f"Finished: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}\n{e}")
        exit(1)



def copy_files(sub, subs, betas_type, chosen_roi, n_latents, alpha_vdvae, alpha_cliptext, alpha_clipvision, dir_where_to_save):
    # Comment or uncomment the following lines according to what you would want to save (without this saving, data of the same subject is overwitten at every run)
    print(f'Copying results into {dir_where_to_save}...')
    # Prepare source directories pathts
    src_processed_dir = processed_dir + f'subj_{sub}/'
    src_extracted_dir = extracted_dir + f'subj_{sub}/'
    src_regression_dir = regression_dir + f'subj_{sub}/'
    src_predicted_dir = predicted_dir + f'subj_{sub}/'
    src_eval_features_dir = eval_features_dir + f'subj_{sub}/'
    src_eval_features_gt_dir = eval_features_dir + f'subj_{sub}/ground_truth/'
    if use_other_model_as_first_step == 0 or use_other_model_as_first_step == 2: #VDVAE
        src_eval_features_vdvae_dir = eval_features_dir + f'subj_{sub}/vdvae/'
    src_results_metrics_dir = results_metrics_dir + f'subj_{sub}/'
    src_results_vdvae_dir = results_vae_dir + f'subj_{sub}/'
    src_results_versatile_dir = results_versatile_dir + f'subj_{sub}/'
    src_results_metrics_dir = results_metrics_dir + f'subj_{sub}/'
    src_results_gt_dir = results_gt_dir + f'subj_{sub}/'
    
    # Prepare destination directories paths
    dst_results_metrics_dir = dir_where_to_save + '/metrics'
    dst_results_vdvae_dir = dir_where_to_save + '/reconstructions-vdvae'
    dst_results_versatile_dir = dir_where_to_save + '/reconstructions-versatile_diffusion'
    dst_processed_dir = dir_where_to_save + '/04_processed_data'
    dst_extracted_dir = dir_where_to_save + '/05_extracted_features'
    dst_regression_dir = dir_where_to_save + '/06_regression_weights'
    dst_predicted_dir = dir_where_to_save + '/07_predicted_features'
    dst_eval_features_dir = dir_where_to_save + '/08_eval_features'
    dst_eval_features_gt_dir = dir_where_to_save + '/08_eval_features/ground_truth'
    dst_eval_features_vdvae_dir = dir_where_to_save + '/08_eval_features/vdvae'
    dst_results_gt_dir = dir_where_to_save + '/reconstructions-gt'
    
    # Create destination directories
    os.makedirs(dst_results_metrics_dir)
    os.makedirs(dst_results_vdvae_dir)
    os.makedirs(dst_results_versatile_dir)
    os.makedirs(dst_results_gt_dir)
    os.makedirs(dst_processed_dir)
    os.makedirs(dst_extracted_dir)
    os.makedirs(dst_regression_dir)
    os.makedirs(dst_predicted_dir)
    os.makedirs(dst_eval_features_dir)
    os.makedirs(dst_eval_features_gt_dir)
    os.makedirs(dst_eval_features_vdvae_dir)
    
    # Add /
    dst_results_metrics_dir += '/'
    dst_results_vdvae_dir += '/'
    dst_results_versatile_dir += '/'
    dst_results_gt_dir += '/'
    dst_processed_dir += '/'
    dst_extracted_dir += '/'
    dst_regression_dir += '/'
    dst_predicted_dir += '/'
    dst_eval_features_dir += '/'
    dst_eval_features_gt_dir += '/'
    dst_eval_features_vdvae_dir += '/'

    # METRICS
    # VDVAE
    '''
    shutil.copy2(src_results_metrics_dir+f'metrics_VDVAElatents_percentage_pmedianae-sub{sub}-alpha{alpha_vdvae}.png',
                 dst_results_metrics_dir+f'metrics_VDVAElatents_percentage_pmedianae-sub{sub}-alpha{alpha_vdvae}.png')
    shutil.copy2(src_results_metrics_dir+f'metrics_VDVAElatents_percentage_scatter-CUT_pmeanae-sub{sub}-alpha{alpha_vdvae}.png',
                 dst_results_metrics_dir+f'metrics_VDVAElatents_percentage_scatter-CUT_pmeanae-sub{sub}-alpha{alpha_vdvae}.png')
    shutil.copy2(src_results_metrics_dir+f'metrics_VDVAElatents_percentage_scatter-CUT_pmedianae-sub{sub}-alpha{alpha_vdvae}.png',
                 dst_results_metrics_dir+f'metrics_VDVAElatents_percentage_scatter-CUT_pmedianae-sub{sub}-alpha{alpha_vdvae}.png')
    shutil.copy2(src_results_metrics_dir+f'metrics_VDVAElatents_percentage_scatter_pmeanae-sub{sub}-alpha{alpha_vdvae}.png',
                 dst_results_metrics_dir+f'metrics_VDVAElatents_percentage_scatter_pmeanae-sub{sub}-alpha{alpha_vdvae}.png')
    shutil.copy2(src_results_metrics_dir+f'metrics_VDVAElatents_percentage_scatter_pmedianae-sub{sub}-alpha{alpha_vdvae}.png',
                 dst_results_metrics_dir+f'metrics_VDVAElatents_percentage_scatter_pmedianae-sub{sub}-alpha{alpha_vdvae}.png')    
    shutil.copy2(src_results_metrics_dir+f'metrics_VDVAElatents-sub{sub}-alpha{alpha_vdvae}.csv',
                 dst_results_metrics_dir+f'metrics_VDVAElatents-sub{sub}-alpha{alpha_vdvae}.csv')
    
    # CLIP-text
    shutil.copy2(src_results_metrics_dir+f'metrics_CLIPtext_percentage_pmedianae-sub{sub}-alpha{alpha_cliptext}.png',
                 dst_results_metrics_dir+f'metrics_CLIPtext_percentage_pmedianae-sub{sub}-alpha{alpha_cliptext}.png')
    shutil.copy2(src_results_metrics_dir+f'metrics_CLIPtext_percentage_scatter-CUT_pmeanae-sub{sub}-alpha{alpha_cliptext}.png',
                 dst_results_metrics_dir+f'metrics_CLIPtext_percentage_scatter-CUT_pmeanae-sub{sub}-alpha{alpha_cliptext}.png')
    shutil.copy2(src_results_metrics_dir+f'metrics_CLIPtext_percentage_scatter-CUT_pmedianae-sub{sub}-alpha{alpha_cliptext}.png',
                 dst_results_metrics_dir+f'metrics_CLIPtext_percentage_scatter-CUT_pmedianae-sub{sub}-alpha{alpha_cliptext}.png')
    shutil.copy2(src_results_metrics_dir+f'metrics_CLIPtext_percentage_scatter_pmeanae-sub{sub}-alpha{alpha_cliptext}.png',
                 dst_results_metrics_dir+f'metrics_CLIPtext_percentage_scatter_pmeanae-sub{sub}-alpha{alpha_cliptext}.png')
    shutil.copy2(src_results_metrics_dir+f'metrics_CLIPtext_percentage_scatter_pmedianae-sub{sub}-alpha{alpha_cliptext}.png',
                 dst_results_metrics_dir+f'metrics_CLIPtext_percentage_scatter_pmedianae-sub{sub}-alpha{alpha_cliptext}.png')    
    shutil.copy2(src_results_metrics_dir+f'metrics_CLIPtext-sub{sub}-alpha{alpha_cliptext}.csv',
                 dst_results_metrics_dir+f'metrics_CLIPtext-sub{sub}-alpha{alpha_cliptext}.csv')
    
    # CLIP-vision
    shutil.copy2(src_results_metrics_dir+f'metrics_CLIPvision_percentage_pmedianae-sub{sub}-alpha{alpha_clipvision}.png',
                 dst_results_metrics_dir+f'metrics_CLIPvision_percentage_pmedianae-sub{sub}-alpha{alpha_clipvision}.png')
    shutil.copy2(src_results_metrics_dir+f'metrics_CLIPvision_percentage_scatter-CUT_pmeanae-sub{sub}-alpha{alpha_clipvision}.png',
                 dst_results_metrics_dir+f'metrics_CLIPvision_percentage_scatter-CUT_pmeanae-sub{sub}-alpha{alpha_clipvision}.png')
    shutil.copy2(src_results_metrics_dir+f'metrics_CLIPvision_percentage_scatter-CUT_pmedianae-sub{sub}-alpha{alpha_clipvision}.png',
                 dst_results_metrics_dir+f'metrics_CLIPvision_percentage_scatter-CUT_pmedianae-sub{sub}-alpha{alpha_clipvision}.png')
    shutil.copy2(src_results_metrics_dir+f'metrics_CLIPvision_percentage_scatter_pmeanae-sub{sub}-alpha{alpha_clipvision}.png',
                 dst_results_metrics_dir+f'metrics_CLIPvision_percentage_scatter_pmeanae-sub{sub}-alpha{alpha_clipvision}.png')
    shutil.copy2(src_results_metrics_dir+f'metrics_CLIPvision_percentage_scatter_pmedianae-sub{sub}-alpha{alpha_clipvision}.png',
                 dst_results_metrics_dir+f'metrics_CLIPvision_percentage_scatter_pmedianae-sub{sub}-alpha{alpha_clipvision}.png')    
    shutil.copy2(src_results_metrics_dir+f'metrics_CLIPvision-sub{sub}-alpha{alpha_clipvision}.csv',
                 dst_results_metrics_dir+f'metrics_CLIPvision-sub{sub}-alpha{alpha_clipvision}.csv')'''
    
    # First reconstruction
    shutil.copy2(src_results_metrics_dir+'paircorr_heatmap_first_reconstruction_alexnet-2.png',
                 dst_results_metrics_dir+'paircorr_heatmap_first_reconstruction_alexnet-2.png')
    shutil.copy2(src_results_metrics_dir+'paircorr_heatmap_first_reconstruction_alexnet-5.png',
                 dst_results_metrics_dir+'paircorr_heatmap_first_reconstruction_alexnet-5.png')
    shutil.copy2(src_results_metrics_dir+'paircorr_heatmap_first_reconstruction_clip-final.png',
                 dst_results_metrics_dir+'paircorr_heatmap_first_reconstruction_clip-final.png')
    shutil.copy2(src_results_metrics_dir+'paircorr_heatmap_first_reconstruction_inceptionv3-avgpool.png',
                 dst_results_metrics_dir+'paircorr_heatmap_first_reconstruction_inceptionv3-avgpool.png')
    shutil.copy2(src_results_metrics_dir+f'metrics_first_reconstruction-sub{sub}.csv',
                 dst_results_metrics_dir+f'metrics_first_reconstruction-sub{sub}.csv')
    
    # Full reconstruction
    shutil.copy2(src_results_metrics_dir+'paircorr_heatmap_alexnet-2.png',
                 dst_results_metrics_dir+'paircorr_heatmap_alexnet-2.png')
    shutil.copy2(src_results_metrics_dir+'paircorr_heatmap_alexnet-5.png',
                 dst_results_metrics_dir+'paircorr_heatmap_alexnet-5.png')
    shutil.copy2(src_results_metrics_dir+'paircorr_heatmap_clip-final.png',
                 dst_results_metrics_dir+'paircorr_heatmap_clip-final.png')
    shutil.copy2(src_results_metrics_dir+'paircorr_heatmap_inceptionv3-avgpool.png',
                 dst_results_metrics_dir+'paircorr_heatmap_inceptionv3-avgpool.png')
    shutil.copy2(src_results_metrics_dir+f'metrics_full_reconstruction-sub{sub}.csv',
                 dst_results_metrics_dir+f'metrics_full_reconstruction-sub{sub}.csv')
    

    
    # IMAGES RECONSTRUCTIONS
    # GROUND TRUTH
    imgs_paths = glob.glob(src_results_gt_dir + '*.png')
    for img_path in imgs_paths:
        shutil.copy2(img_path,
                     dst_results_gt_dir+os.path.basename(img_path))
    # VDVAE
    imgs_paths = glob.glob(src_results_vdvae_dir + '*.png')
    for img_path in imgs_paths:
        shutil.copy2(img_path,
                     dst_results_vdvae_dir+os.path.basename(img_path))
        
    # Versatile Diffusion
    imgs_paths = glob.glob(src_results_versatile_dir + '*.png')
    for img_path in imgs_paths:
        shutil.copy2(img_path,
                     dst_results_versatile_dir+os.path.basename(img_path))
   
    # 04: PROCESSED DATA
    '''shutil.copy2(src_processed_dir+'annotations_train.npy',
                 dst_processed_dir+'annotations_train.npy')
    shutil.copy2(src_processed_dir+'annotations_test.npy',
                 dst_processed_dir+'annotations_test.npy')
    shutil.copy2(src_processed_dir+'betas_train.npy',
                 dst_processed_dir+'betas_train.npy')
    shutil.copy2(src_processed_dir+'betas_test.npy',
                 dst_processed_dir+'betas_test.npy')
    shutil.copy2(src_processed_dir+'stimuli_train.npy',
                 dst_processed_dir+'stimuli_train.npy')
    shutil.copy2(src_processed_dir+'stimuli_test.npy',
                 dst_processed_dir+'stimuli_test.npy')
    shutil.copy2(src_processed_dir+'stimuli_labels_train.npy',
                 dst_processed_dir+'stimuli_labels_train.npy')
    shutil.copy2(src_processed_dir+'stimuli_labels_test.npy',
                 dst_processed_dir+'stimuli_labels_test.npy')'''
    
    # 05: EXTRACTED FEATURES
    '''shutil.copy2(src_extracted_dir+'vdvae_latents.npz',
                 dst_extracted_dir+'vdvae_latents.npz')
    shutil.copy2(src_extracted_dir+'cliptext_train.npy',
                 dst_extracted_dir+'cliptext_train.npy')
    shutil.copy2(src_extracted_dir+'cliptext_test.npy',
                 dst_extracted_dir+'cliptext_test.npy')
    shutil.copy2(src_extracted_dir+'clipvision_train.npy',
                 dst_extracted_dir+'clipvision_train.npy')
    shutil.copy2(src_extracted_dir+'clipvision_test.npy',
                 dst_extracted_dir+'clipvision_test.npy')'''
    
    # 06: REGRESSION WEIGHTS
    '''shutil.copy2(src_regression_dir+'vdvae_regression_weights.pkl',
                 dst_regression_dir+'vdvae_regression_weights.pkl')
    shutil.copy2(src_regression_dir+f'cliptext_regression_weights_alpha{alpha_cliptext}.pkl',
                 dst_regression_dir+f'cliptext_regression_weights_alpha{alpha_cliptext}.pkl')
    shutil.copy2(src_regression_dir+f'clipvision_regression_weights_alpha{alpha_clipvision}.pkl',
                 dst_regression_dir+f'clipvision_regression_weights_alpha{alpha_clipvision}.pkl')
    #best alphas for ridge
    shutil.copy2(src_regression_dir+'vdvae_best_alpha.npy',
                 dst_regression_dir+'vdvae_best_alpha.npy')
    shutil.copy2(src_regression_dir+'cliptext_best_alphas.npy',
                 dst_regression_dir+'cliptext_best_alphas.npy')
    shutil.copy2(src_regression_dir+'clipvision_best_alphas.npy',
                 dst_regression_dir+'clipvision_best_alphas.npy')'''
    
    # 07: PREDICTED FEATURES
    '''shutil.copy2(src_predicted_dir+'vdvae_pred.npy',
                 dst_predicted_dir+'vdvae_pred.npy')
    shutil.copy2(src_predicted_dir+'vdvae_pred_nozscore_formetrics.npy',
                 dst_predicted_dir+'vdvae_pred_nozscore_formetrics.npy')
    shutil.copy2(src_predicted_dir+'cliptext_pred.npy',
                 dst_predicted_dir+'cliptext_pred.npy')
    shutil.copy2(src_predicted_dir+'cliptext_pred_nozscore_formetrics.npy',
                 dst_predicted_dir+'cliptext_pred_nozscore_formetrics.npy')
    shutil.copy2(src_predicted_dir+'clipvision_pred.npy',
                 dst_predicted_dir+'clipvision_pred.npy')
    shutil.copy2(src_predicted_dir+'clipvision_pred_nozscore_formetrics.npy',
                 dst_predicted_dir+'clipvision_pred_nozscore_formetrics.npy')'''
    
    # 08: EVAULATION FEATURES
    '''# Generated
    shutil.copy2(src_eval_features_dir+'alexnet_2.npy',
                 dst_eval_features_dir+'alexnet_2.npy')
    shutil.copy2(src_eval_features_dir+'alexnet_5.npy',
                 dst_eval_features_dir+'alexnet_5.npy')
    shutil.copy2(src_eval_features_dir+'clip_final.npy',
                 dst_eval_features_dir+'clip_final.npy')
    shutil.copy2(src_eval_features_dir+'efficientnet_avgpool.npy',
                 dst_eval_features_dir+'efficientnet_avgpool.npy')
    shutil.copy2(src_eval_features_dir+'inceptionv3_avgpool.npy',
                 dst_eval_features_dir+'inceptionv3_avgpool.npy')
    shutil.copy2(src_eval_features_dir+'swav_avgpool.npy',
                 dst_eval_features_dir+'swav_avgpool.npy')
    
    # Ground truth
    shutil.copy2(src_eval_features_gt_dir+'alexnet_2.npy',
                 dst_eval_features_gt_dir+'alexnet_2.npy')
    shutil.copy2(src_eval_features_gt_dir+'alexnet_5.npy',
                 dst_eval_features_gt_dir+'alexnet_5.npy')
    shutil.copy2(src_eval_features_gt_dir+'clip_final.npy',
                 dst_eval_features_gt_dir+'clip_final.npy')
    shutil.copy2(src_eval_features_gt_dir+'efficientnet_avgpool.npy',
                 dst_eval_features_gt_dir+'efficientnet_avgpool.npy')
    shutil.copy2(src_eval_features_gt_dir+'inceptionv3_avgpool.npy',
                 dst_eval_features_gt_dir+'inceptionv3_avgpool.npy')
    shutil.copy2(src_eval_features_gt_dir+'swav_avgpool.npy',
                 dst_eval_features_gt_dir+'swav_avgpool.npy')
    
    # Vdvae generated
    shutil.copy2(src_eval_features_vdvae_dir+'alexnet_2.npy',
                 dst_eval_features_vdvae_dir+'alexnet_2.npy')
    shutil.copy2(src_eval_features_vdvae_dir+'alexnet_5.npy',
                 dst_eval_features_vdvae_dir+'alexnet_5.npy')
    shutil.copy2(src_eval_features_vdvae_dir+'clip_final.npy',
                 dst_eval_features_vdvae_dir+'clip_final.npy')
    shutil.copy2(src_eval_features_vdvae_dir+'efficientnet_avgpool.npy',
                 dst_eval_features_vdvae_dir+'efficientnet_avgpool.npy')
    shutil.copy2(src_eval_features_vdvae_dir+'inceptionv3_avgpool.npy',
                 dst_eval_features_vdvae_dir+'inceptionv3_avgpool.npy')
    shutil.copy2(src_eval_features_vdvae_dir+'swav_avgpool.npy',
                 dst_eval_features_vdvae_dir+'swav_avgpool.npy')'''
    
    print('Results copied.')
    
    # Prepare content
    content = f"""
    sub = {sub} #[1, 2, 5, 7]
    betas_type = {betas_type} #type of betas used: '' is the only current option
    chosen_roi = {chosen_roi} #chosen ROI: nsdgeneral
    n_latents = {n_latents} # Number of latent variables of the VDVAE to be predicted. We found that for GRU-model, 15 is better then 31
    alpha_vdvae = {alpha_vdvae} # Regularization factor for Ridge of VDVAE (for the case in which GRU-model is not used)
    alpha_cliptext = {alpha_cliptext} # Regularization factor for Ridge of CLIP-text
    alpha_clipvision = {alpha_clipvision} # Regularization factor for Ridge of CLIP-vision
    use_other_model_as_first_step = {use_other_model_as_first_step} # 0: ridge on vdvae (as in brain-diffuser by Furkan Ozcelik and Rufin VanRullen), 2: GRU on vdvae
    use_other_model_for_clip = {use_other_model_for_clip} # 0: Ridge, 1: NN. Our results demonstrated that Ridge works better for CLIP predictions
    normalize_pred_vdvae = {normalize_pred_vdvae} # Normalize predicted VDVAE embeddings of the test set. NOTE: for now it is implemented only for NN: elsewhere it's always 1
    normalize_pred_clip = {normalize_pred_clip} # Normalize predicted CLIP embeddings of the test set. NOTE: for now it is implemented only for NN: elsewhere it's always 1
    """
    
    # Save to text file
    with open(dir_where_to_save+'/parameters.txt', "w") as file:
        file.write(content)
    
    print('Parameters saved.')




def run_full_pipeline(sub, subs, betas_type, chosen_roi, n_latents, alpha_vdvae, alpha_cliptext, alpha_clipvision, save_results=True, personalized_foldername=None):
    # Data preparation
    run_command(f'cd data && python prepare_nsddata.py -sub={sub} -chosen_roi={chosen_roi} -betas_type={betas_type}')

    # First-stage reconstruction
    if use_other_model_as_first_step == 0 or use_other_model_as_first_step == 2: # VDVAE ridge or VDVAE NN
        run_command(f'python scripts/01_vdvae_extract_features.py -sub={sub}')
        if use_other_model_as_first_step == 0: #ridge
            run_command(f'python scripts/02_vdvae_regression.py -sub={sub} -n_latents={n_latents} -alpha={alpha_vdvae}')
        elif use_other_model_as_first_step == 2: #NN
            run_command(f'python scripts/02_vdvae_regression_nn.py -sub={sub} -n_latents={n_latents} -alpha={alpha_vdvae} -normalize_pred={normalize_pred_vdvae}')
        else:
            raise ValueError("Wrong use_other_model_as_first_step!!!")
        run_command(f'python scripts/03_vdvae_reconstruct_images.py -sub={sub} -n_latents={n_latents}')
    
    # CLIP
    run_command(f'python scripts/04_cliptext_extract_features.py -sub={sub}')
    run_command(f'python scripts/05_clipvision_extract_features.py -sub={sub}')
    if use_other_model_for_clip == 0:
        run_command(f'python scripts/06_cliptext_regression.py -sub={sub} -alpha={alpha_cliptext}')
        run_command(f'python scripts/07_clipvision_regression.py -sub={sub} -alpha={alpha_clipvision}')
    elif use_other_model_for_clip == 1:
        run_command(f'python scripts/06_cliptext_regression_nn.py -sub={sub} -normalize_pred={normalize_pred_clip}')
        run_command(f'python scripts/07_clipvision_regression_nn.py -sub={sub} -normalize_pred={normalize_pred_clip}')
    
    # Versatile Diffusion
    run_command(f'python scripts/08_versatilediffusion_reconstruct_images.py -sub={sub} -use_other_model_as_first_step={use_other_model_as_first_step}')
    
    # Evaluation
    run_command(f'python scripts/09_save_test_images.py -sub={sub}')
    run_command(f'python scripts/10_eval_extract_features.py -sub={sub} -is_gt=1')
    run_command(f'python scripts/10_eval_extract_features.py -sub={sub}')
    run_command(f'python scripts/10_eval_extract_features_vdvae.py -sub={sub} -use_other_model_as_first_step={use_other_model_as_first_step}')
    run_command(f'python scripts/11_evaluate_reconstruction.py -sub={sub}')
    run_command(f'python scripts/11_evaluate_reconstruction_vdvae.py -sub={sub} -use_other_model_as_first_step={use_other_model_as_first_step}')
    
    run_command(f'python scripts/12_print_summary_metrics.py -sub={sub}')
    
    # Save results
    if not save_results: return
    # Get the current date and time
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M") #year-month-day_hour-minute
    if personalized_foldername == None:
        dir_where_to_save = formatted_datetime + f'-sub{sub}-{betas_type}-{chosen_roi}-lat{n_latents}-avdvae{alpha_vdvae}-actxt{alpha_cliptext}-acvis{alpha_clipvision}-uomafs{use_other_model_as_first_step}'   
    else:
        dir_where_to_save = personalized_foldername
    # Create the folder
    if not os.path.exists(saved_dir + dir_where_to_save):
        os.makedirs(saved_dir + dir_where_to_save)
    else:
        raise Exception(f'The directory {dir_where_to_save} alread exits')
    # Copy the files
    copy_files(sub, subs, betas_type, chosen_roi, n_latents, alpha_vdvae, alpha_cliptext, alpha_clipvision, saved_dir + dir_where_to_save)



def run_all_subjects(n_latents, subjs_list=[1, 2, 5, 7]):
    for curr_subj in subjs_list:
        run_full_pipeline(curr_subj, n_latents, alpha_vdvae, alpha_cliptext, alpha_clipvision)


def main():
    run_full_pipeline(sub, subs, betas_type, chosen_roi, n_latents, alpha_vdvae, alpha_cliptext, alpha_clipvision)

    
if __name__ == "__main__":
    main()