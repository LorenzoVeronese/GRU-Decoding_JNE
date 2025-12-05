import sys
import numpy as np
import sklearn.linear_model as skl
import pickle
import argparse
import os
from utils import metrics_vector_compute
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-normalize_pred", "--normalize_pred",help="Normalize predictions", default=1)
args = parser.parse_args()
sub=int(args.sub)
normalize_pred = int(args.normalize_pred)



##### SET PATHS
base_path = ''
processed_dir = base_path + 'data/04_processed_data/'
extracted_dir = base_path + 'data/05_extracted_features/'
regression_dir = base_path + 'data/06_regression_weights/subj_{}/'.format(sub)
predicted_dir = base_path + 'data/07_predicted_features/subj_{}/'.format(sub)
results_metrics_dir = base_path + 'results/metrics/'
if not os.path.exists(results_metrics_dir+f'subj_{sub}'):
    os.makedirs(results_metrics_dir+f'subj_{sub}')


##### DEFINING THE MODEL
class FMRIToCLIPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads, dropout_prob=0.5):
        super(FMRIToCLIPModel, self).__init__()
        self.lstm1 = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.ln_lstm1 = nn.LayerNorm(hidden_size*2)
        self.dropout = nn.Dropout(dropout_prob)
        self.lstm2 = nn.GRU(hidden_size*2, hidden_size, batch_first=True, bidirectional=True)
        self.ln_lstm2 = nn.LayerNorm(hidden_size*2)
        self.fc_final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size*2, hidden_size*2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size*2, output_size[0] * output_size[1])
        )
        self.output_size = output_size

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x, _ = self.lstm1(x)
        x = self.ln_lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.ln_lstm2(x)
        x = self.fc_final(x)
        return x.view(-1, self.output_size[0], self.output_size[1])



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



##### TRAIN
print()
print("##############################")
print('Starting the training (NN)...')
# Prepare data
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = TensorDataset(torch.tensor(train_betas), torch.tensor(train_clip))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(torch.tensor(test_betas), torch.tensor(test_clip))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define model, loss function, and optimizer
input_size = train_betas.shape[1]
hidden_size = 256
output_size = train_clip.shape[1:]  # This should be (num_embed, num_dim)

model = FMRIToCLIPModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_heads=4, dropout_prob=0.5).to(device)

criterion = nn.MSELoss()
optimizer = optim.NAdam(model.parameters(), lr=0.001, weight_decay=5e-6)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=6, factor=0.5, verbose=True)

# Training loop
num_epochs = 500
best_val_loss = float('inf')
early_stopping_counter = 0
early_stopping_patience = 12

train_mse_losses = []
train_mae_losses = []
val_mse_losses = []
val_mae_losses = []
learning_rates = [] 

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_train_mse = 0.0
    running_train_mae = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, targets.float()) 
        loss.backward()
        optimizer.step()
        
        running_train_mse += loss.item() * inputs.size(0)
        running_train_mae += torch.mean(torch.abs(outputs - targets.float())).item() * inputs.size(0)
    
    # Validation phase
    model.eval()
    running_val_mse = 0.0
    running_val_mae = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.float())
            mse_loss = torch.mean((outputs - targets.float()) ** 2)
            mae_loss = torch.mean(torch.abs(outputs - targets.float()))
            running_val_mse += mse_loss.item() * inputs.size(0)
            running_val_mae += mae_loss.item() * inputs.size(0)
    
    # Compute average losses
    train_mse = running_train_mse / len(train_loader.dataset)
    train_mae = running_train_mae / len(train_loader.dataset)
    val_mse = running_val_mse / len(val_loader.dataset)
    val_mae = running_val_mae / len(val_loader.dataset)
    
    train_mse_losses.append(train_mse)
    train_mae_losses.append(train_mae)
    val_mse_losses.append(val_mse)
    val_mae_losses.append(val_mae)

    # Learning rate scheduling
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr) 
    scheduler.step(val_mse) 

    # Early stopping check
    if val_mse < best_val_loss:
        best_val_loss = val_mse
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train MSE: {train_mse:.5f}, Train MAE: {train_mae:.5f}, "
          f"Val MSE: {val_mse:.5f}, Val MAE: {val_mae:.5f}, LR: {current_lr:.6f}")

# Save the model
torch.save(model.state_dict(), regression_dir+'cliptext_nn.pth')



##### PLOT
print()
print("##############################")
print('Plotting the losses...')
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_mse_losses, label='Train MSE')
plt.plot(val_mse_losses, label='Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training and Validation MSE Losses')

plt.subplot(1, 3, 2)
plt.plot(train_mae_losses, label='Train MAE')
plt.plot(val_mae_losses, label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE Loss')
plt.legend()
plt.title('Training and Validation MAE Losses')

plt.subplot(1, 3, 3)
plt.plot(learning_rates)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate over Epochs')

plt.tight_layout()

plt.savefig(results_metrics_dir+f'subj_{sub}/'.format(sub) + 'training_nn_cliptext.png')

plt.close()



##### TEST
test_dataset = TensorDataset(torch.tensor(test_betas), torch.tensor(test_clip))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.eval()

# Evaluate on test set
test_mse = 0.0
test_mae = 0.0
test_mse_losses = []
test_mae_losses = []
test_predictions = []

# Evaluation loop
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs.float())
        mse_loss = torch.mean((outputs - targets.float()) ** 2)
        mae_loss = torch.mean(torch.abs(outputs - targets.float()))

        # Accumulate the total losses for MSE and MAE
        test_mse += mse_loss.item() * inputs.size(0)
        test_mae += mae_loss.item() * inputs.size(0)

        # Store individual losses for standard deviation computation
        test_mse_losses.append(mse_loss.item())
        test_mae_losses.append(mae_loss.item())

        # Collect predictions
        test_predictions.append(outputs.cpu().numpy())

# Compute average test MSE and MAE
test_mse /= len(test_loader.dataset)
test_mae /= len(test_loader.dataset)

# Compute standard deviations of MSE and MAE
test_mse_std = np.std(test_mse_losses)
test_mae_std = np.std(test_mae_losses)


content = f"""test_mse={test_mse},
test_mse_std={test_mse_std},
test_mae={test_mae},
test_mae_std={test_mae_std}"""

with open(results_metrics_dir+f'subj_{sub}/mse_mae_cliptext_nn.txt', "w") as file:
    file.write(content)






##### SAVE PREDICTIONS
# Concatenate all batches of predictions
test_predictions = np.concatenate(test_predictions, axis=0)

# Normalize
if normalize_pred == 1:
    std_norm_test_clips = (test_predictions - np.mean(test_predictions,axis=0)) / np.std(test_predictions,axis=0)
    test_predictions = std_norm_test_clips * np.std(train_clip,axis=0) + np.mean(train_clip,axis=0)

# Save predictions
os.makedirs(predicted_dir, exist_ok=True)
np.save(predicted_dir + 'cliptext_pred.npy', test_predictions)

print("Test set predictions completed and saved.")

# Optional: Calculate and print some statistics about the predictions
print("Shape of test predictions:", test_predictions.shape)
print("Mean of test predictions:", np.mean(test_predictions))
print("Std of test predictions:", np.std(test_predictions))
print("Min of test predictions:", np.min(test_predictions))
print("Max of test predictions:", np.max(test_predictions))