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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import time
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-alpha", "--alpha",help="Regularization of Ridge", default=50000)
parser.add_argument("-n_latents", "--n_latents", help="Number of latent variables", default=31)
parser.add_argument("-normalize_pred", "--normalize_pred",help="Normalize predictions", default=1)
args = parser.parse_args()
sub=int(args.sub)
alpha = int(args.alpha)
n_latents = int(args.n_latents)
normalize_pred = int(args.normalize_pred)
assert n_latents in range(1, 32)
compute_cost_metrics = 0 #Set to 1 if want to compute performances reported in the paper (Training time, inference time...)



##### SET PATHS
base_path = '' 
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



##### GET BETAS
print()
print("##############################")
print('Retrieving betas...')
train_betas = np.load(processed_dir + 'betas_train.npy')
test_betas = np.load(processed_dir + 'betas_test.npy')



##### PREPROCESS (z-score)
print()
print("##############################")
print('Z-score normalization...')

epsilon = 1e-10
norm_mean_train = np.mean(train_betas, axis=0)
norm_std_train = np.std(train_betas, axis=0, ddof=1)


train_betas = (train_betas - norm_mean_train) / (norm_std_train+epsilon)
test_betas = (test_betas - norm_mean_train) / (norm_std_train+epsilon)

print("    Mean and standard deviation of training betas: ", np.mean(train_betas), np.std(train_betas))
print("    Mean and standard deviation of test betas: ", np.mean(test_betas), np.std(test_betas))
print("    Max and min of training betas: ", np.max(train_betas),np.min(train_betas))
print("    Max and min of test betas: ", np.max(test_betas),np.min(test_betas))


##### VALIDATION SET (for initial training: to find best learning rates and number of epochs)
seed = 42
np.random.seed(seed)

num_samples = train_betas.shape[0]
perm_index = np.random.permutation(num_samples)

train_fmri = train_betas[perm_index]
train_latents = train_latents[perm_index]

num_val = int(num_samples * 0.10)
val_fmri = train_fmri[num_samples-num_val:]
train_fmri = train_fmri[:num_samples-num_val]
val_latents = train_latents[num_samples-num_val:]
train_latents = train_latents[:num_samples-num_val]

print("Train set length: ", train_fmri.shape[0])
print("Validation set length: ", val_fmri.shape[0])



from torchvision import transforms

norm_mean_train = np.mean(train_betas, axis=0)
norm_scale_train = np.std(train_betas, axis=0, ddof=1)

def z_score_back(sample):
    return sample * norm_scale_train + norm_mean_train

def z_score(sample):
    return (sample - norm_mean_train) * norm_scale_train

class GaussianNoise:
    def __init__(self, mean, std_dev, apply_probab):
        self.mean = mean
        self.std_dev = std_dev
        self.apply_probab = apply_probab

    def __call__(self, sample):
        if random.random() < self.apply_probab:
            sample = z_score_back(sample)
            noise = np.random.randn(len(sample)) * self.std_dev + self.mean
            sample= sample + noise
            sample = z_score(sample)
        return sample

class RandomMask:
    def __init__(self, max_region_len, apply_probab):
        self.max_region_len = max_region_len
        self.apply_probab = apply_probab

    def __call__(self, sample):
        if random.random() < self.apply_probab:
            region_len = np.random.randint(1, self.max_region_len)
            start_idx = np.random.randint(0, len(sample)-region_len)

            sample[start_idx:start_idx+region_len] = 0
        return sample

# Define the transformations
apply_probab = 0.3
data_transforms = transforms.Compose([
    GaussianNoise(mean=0, std_dev=1, apply_probab=apply_probab),
    RandomMask(max_region_len=500, apply_probab=apply_probab),
])

device='cuda'


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0):
        super(GRUModel, self).__init__()

        self.lstm1 = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True) #(N, L, H)
        self.ln_lstm1 = nn.LayerNorm(hidden_size*2)

        self.dropout_lstm_1_2 = nn.Dropout(dropout_prob)

        self.lstm2 = nn.GRU(hidden_size*2, hidden_size, batch_first=True, bidirectional=True)
        self.ln_lstm2 = nn.LayerNorm(hidden_size*2)

        self.fc_final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size*2, hidden_size*2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size*2, output_size),

        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1) #get dummy dimension at 2nd place

        x, _ = self.lstm1(x)
        x = self.ln_lstm1(x)

        x = self.dropout_lstm_1_2(x)
        x, _ = self.lstm2(x)
        x = self.ln_lstm2(x)

        x = self.fc_final(x)

        return x
        



##### TRAIN MODEL WITH VALIDATION
print()
print("##############################")
print('Training the model with validation set.')
# Define your dataset
batch_size=64
train_fmri=train_fmri
train_latents=train_latents

train_dataset = TensorDataset(torch.tensor(train_fmri), torch.tensor(train_latents)) #[0:128]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Define your model, loss function, and optimizer
input_size = train_fmri.shape[1]
hidden_size = 100 
output_size = train_latents.shape[1]

model = GRUModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dropout_prob=0.5).to(device)

criterion = nn.MSELoss()
optimizer = optim.NAdam(model.parameters(), lr=0.001, weight_decay=5e-6)

# Validation dataset and loader
val_dataset = TensorDataset(torch.tensor(val_fmri), torch.tensor(val_latents))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define criterion for MSE and MAE
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()

# Additional controllers
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=6, factor=0.5, verbose=True)

# early stopping
best_val_loss = float('inf')
early_stopping_counter = 0
early_stopping_patience = 12 

# Training loop
num_epochs = 200 # Early stopping will stop earlier
train_mse_list = []
train_mae_list = []
val_mse_list = []
val_mae_list = []
learning_rates = []
if compute_cost_metrics == 1:
    start_total = time.time()
    torch.cuda.reset_peak_memory_stats(device)

for epoch in range(num_epochs):
    # Training phase
    model.train() 
    running_train_mse = 0.0
    running_train_mae = 0.0
    for inputs, targets in train_loader:
        inputs = torch.stack([data_transforms(sample) for sample in inputs])
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad() 
        outputs = model(inputs.float())
        loss_mse = criterion_mse(outputs, targets.float())
        loss_mae = criterion_mae(outputs, targets.float())
        loss = loss_mse 
        loss.backward() 
        optimizer.step() 
        running_train_mse += loss_mse.item() * inputs.size(0)
        running_train_mae += loss_mae.item() * inputs.size(0)
    
    # Validation phase
    model.eval()
    running_val_mse = 0.0
    running_val_mae = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.float())
            loss_mse = criterion_mse(outputs, targets.float())
            loss_mae = criterion_mae(outputs, targets.float())
            running_val_mse += loss_mse.item() * inputs.size(0)
            running_val_mae += loss_mae.item() * inputs.size(0)
    
    # Compute average losses
    train_mse = running_train_mse / len(train_loader.dataset)
    train_mae = running_train_mae / len(train_loader.dataset)
    train_mse_list.append(train_mse)
    train_mae_list.append(train_mae)

    val_mse = running_val_mse / len(val_loader.dataset)
    val_mae = running_val_mae / len(val_loader.dataset)
    val_mse_list.append(val_mse)
    val_mae_list.append(val_mae)

    
    # check if have to reduce lr
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    scheduler.step(val_mse)

    # check early stopping
    if val_mse < best_val_loss:
        best_val_loss = val_mse
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch}')
            break
    print(f"Epoch [{epoch+1}/{num_epochs}], Train MSE: {train_mse:.5f}, Train MAE: {train_mae:.5f}, Val MSE: {val_mse:.5f}, Val MAE: {val_mae:.5f}, LR: {current_lr:.6f}")

if compute_cost_metrics == 1:
    end_total = time.time()  # End total training time
    total_training_time = end_total - start_total
    print(f"Total training time: {total_training_time:.2f} seconds")

    max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert bytes to MB
    print(f"\nPeak GPU memory allocated during training: {max_mem:.2f} MB")




##### TRAIN MODEL WITH FULL DATASET (with selected number of batches and selected learning rate curve)
print()
print("##############################")
print('Training the final model (no validation set).')
num_epochs_final = len(learning_rates)
print('Learning rate schedule:', learning_rates)
print('Number of epochs:', num_epochs_final)
# Reset the model
model = GRUModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dropout_prob=0.5).to(device)

#Merge training and validation sets
full_fmri = np.concatenate((train_fmri, val_fmri), axis=0)
full_latents = np.concatenate((train_latents, val_latents), axis=0)

# Create a new dataset and DataLoader
full_dataset = TensorDataset(torch.tensor(full_fmri), torch.tensor(full_latents))
full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

# Reset optimizer and scheduler
optimizer = optim.NAdam(model.parameters(), lr=0.001, weight_decay=5e-6)

for epoch in range(num_epochs_final):
    # Set the learning rate dynamically based on recorded schedule
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rates[epoch]  # Replay the learning rate

    model.train()
    running_train_mse = 0.0
    for inputs, targets in full_loader:
        inputs = torch.stack([data_transforms(sample) for sample in inputs])
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion_mse(outputs, targets.float())
        loss.backward()
        optimizer.step()

        running_train_mse += loss.item() * inputs.size(0)

    train_mse = running_train_mse / len(full_loader.dataset)
    print(f"Full Training Epoch [{epoch+1}/{num_epochs_final}], Train MSE: {train_mse:.5f}, LR: {learning_rates[epoch]:.6f}")



if compute_cost_metrics == 1:
    model_flops = ModelWithTransformer(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        dropout_prob=0.5
    ).to(device)

    # Switch to eval mode (recommended for profiling)
    model_flops.eval()

    def input_constructor(input_res):

        return {'x': torch.randn(1, *input_res).to(device)}

    dummy_input_shape = (input_size,)  # Only the feature dimension

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model_flops,
            dummy_input_shape,
            as_strings=True,
            print_per_layer_stat=True,
            input_constructor=input_constructor
        )
    print(f"FLOPs (MACs): {macs}")
    print(f"Number of parameters (for gpflops): {params}")

    total_params = sum(p.numel() for p in model_flops.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")



##### TESTING
#model = torch.load('mlp_model.pth')
test_fmri = test_betas
test_dataset = TensorDataset(torch.tensor(test_fmri), torch.tensor(test_latents))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) #DA CAMBIARE con batch_size

if compute_cost_metrics == 1:
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    inference_times = []
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = model(inputs.float())
        torch.cuda.synchronize()
        end = time.perf_counter()
        inference_times.append(end - start)

    mean_inference_time = np.mean(inference_times)
    print(f"Mean inference time: {mean_inference_time:.6f} seconds")

    max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert bytes to MB
    print(f"\nPeak GPU memory allocated during inference: {max_mem:.2f} MB")

# Model evaluation
model.eval()

mse_errors = []
mae_errors = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs.float())
        
        # Compute MSE and MAE for each sample in the batch
        mse = criterion_mse(outputs, targets.float()).item()
        mae = criterion_mae(outputs, targets.float()).item()
        
        # Extend the lists with individual errors
        mse_errors.extend([mse] * inputs.size(0))
        mae_errors.extend([mae] * inputs.size(0))

# Compute average and standard deviation
test_mse = np.mean(mse_errors)
test_mae = np.mean(mae_errors)
test_mse_std = np.std(mse_errors)
test_mae_std = np.std(mae_errors)

content = f"""final train_mse={train_mse_list[-10:]}
final train_mae={train_mae_list[-10:]}
final val_mse={val_mse_list[-10:]}
final val_mae={val_mae_list[-10:]}

test_mse={test_mse},
test_mse_std={test_mse_std},
test_mae={test_mae},
test_mae_std={test_mae_std}"""

if not os.path.exists(results_metrics_dir+f'subj_{sub}/'):
    os.makedirs(results_metrics_dir+f'subj_{sub}/')
with open(results_metrics_dir+f'subj_{sub}/mse_mae_vdvae_nn.txt', "w") as file:
    file.write(content)



import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(train_mse_list[1:], label='Train MSE')
plt.plot(val_mse_list[1:], label='Val MSE')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation MSE')
plt.legend()
plt.tight_layout()
plt.savefig(results_metrics_dir+f'subj_{sub}/mse_plot_vdvae_nn.png')
plt.close()

# Create and save MAE plot
plt.figure(figsize=(10, 6))
plt.plot(train_mae_list[1:], label='Train MAE')
plt.plot(val_mae_list[1:], label='Val MAE')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation MAE')
plt.legend()
plt.tight_layout()
plt.savefig(results_metrics_dir+f'subj_{sub}/mae_plot_vdvae_nn.png')
plt.close()

if not os.path.exists(regression_dir):
    os.makedirs(regression_dir)
torch.save(model, regression_dir + 'full_2lstm_2ln_bn_mlp.pth')



##### SAVE PREDICTIONS
gpu_test_fmri = torch.tensor(test_fmri).float().to(device)
pred_latents = model(gpu_test_fmri)

pred_latents = pred_latents.cpu().detach().numpy()



##### COMPUTE METRICS (on the un-normalized)
metrics_vector_compute.compute_metrics(test_latents, pred_latents, 0, results_metrics_dir,'VDVAElatents', sub, alpha)


# "REVERSED" Z-SCORE
if normalize_pred == 1:
    std_norm_test_latent = (pred_latents - np.mean(pred_latents,axis=0)) / np.std(pred_latents,axis=0)
    pred_latents = std_norm_test_latent * np.std(train_latents,axis=0) + np.mean(train_latents,axis=0)


##### COMPUTE METRICS (on the normalized)
metrics_vector_compute.compute_metrics(test_latents, pred_latents, 0, results_metrics_dir,'VDVAElatentsZscoreInverted', sub, alpha)

if not os.path.exists(predicted_dir):
    os.makedirs(predicted_dir)
np.save(predicted_dir+'vdvae_pred.npy', pred_latents) # 10 sessions => 11.468934634 GB
