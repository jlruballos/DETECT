import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CorruptionAwareVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super(CorruptionAwareVAE, self).__init__()
        
        # Encoder with layer normalization for stability
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder with layer normalization
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def encode(self, x_tilde, mask):
        # Concatenate data and mask
        input_combined = torch.cat([x_tilde, mask], dim=1)
        h = self.encoder(input_combined)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        # Add numerical stability
        std = torch.exp(0.5 * torch.clamp(logvar, min=-10, max=10))
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, mask):
        # Concatenate latent vector and mask
        input_combined = torch.cat([z, mask], dim=1)
        return self.decoder(input_combined)
    
    def forward(self, x_tilde, mask):
        mu, logvar = self.encode(x_tilde, mask)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, mask)
        return x_hat, mu, logvar
    
    def loss_function(self, x, x_hat, mu, logvar, mask):
        # Calculate reconstruction loss with safety checks
        missing_mask = (1 - mask).float()
        n_missing = torch.sum(missing_mask) + 1e-8  # avoid division by zero
        
        # MSE loss only on missing values
        recon_loss = torch.sum(torch.pow((x_hat * missing_mask) - (x * missing_mask), 2)) / n_missing
        
        # KL divergence with numerical stability
        kl_loss = -0.5 * torch.mean(1 + torch.clamp(logvar, min=-10, max=10) - 
                                    mu.pow(2) - torch.exp(torch.clamp(logvar, min=-10, max=10)))
        
        # Weight KL divergence lower to prioritize reconstruction
        beta = 0.1
        return recon_loss + beta * kl_loss, recon_loss, kl_loss

def train_vae(model, dataloader, optimizer, epochs=50, device=device):
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for x, x_tilde, mask in dataloader:
            x = x.to(device)
            x_tilde = x_tilde.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            
            x_hat, mu, logvar = model(x_tilde, mask)
            loss, recon_loss, kl_loss = model.loss_function(x, x_hat, mu, logvar, mask)
            
            # Skip bad batches
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"Warning: NaN or Inf loss detected, skipping batch")
                continue
                
            loss.backward()
            
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Loss: NaN - all batches skipped")

def prepare_dataloader(data, masks, batch_size=64):
    # Properly handle NaN values and normalize
    scaler = StandardScaler()
    
    # Create a copy to avoid modifying original data
    clean_data = np.copy(data)
    
    # For each feature column
    for i in range(data.shape[1]):
        # Get non-missing values for this column
        valid_idx = np.where(masks[:, i] == 1)[0]
        if len(valid_idx) > 0:
            valid_values = data[valid_idx, i]
            
            # Fit scaler on non-missing values
            mean = np.mean(valid_values)
            std = np.std(valid_values) if np.std(valid_values) > 0 else 1.0
            
            # Transform all values (scaling)
            clean_data[:, i] = (data[:, i] - mean) / std
            
    # Replace NaN values with zeros after scaling
    clean_data = np.nan_to_num(clean_data, nan=0.0)
    
    # Create PyTorch tensors
    x = torch.tensor(clean_data, dtype=torch.float32)
    mask = torch.tensor(masks, dtype=torch.float32)
    x_tilde = x.clone()
    
    # Create dataset and dataloader
    dataset = TensorDataset(x, x_tilde, mask)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def impute_subject_data(df, input_columns, epochs=30, device=device):
    """
    Imputes missing values in the dataframe using a corruption-aware VAE model.
    
    This implementation follows the approach described in "VAEs in the Presence of Missing Data"
    by Collier et al., which models missing data as a corruption process and uses a
    conditional VAE where both encoder and decoder have access to the missingness mask.
    
    Args:
        df: Pandas DataFrame with missing values
        input_columns: List of column names to impute
        epochs: Number of training epochs
        device: Computing device ('cuda' or 'cpu')
        
    Returns:
        DataFrame with imputed values
    """
    # Extract data and create mask
    data = df[input_columns].values.astype(np.float32)
    mask = ~np.isnan(data)  # True for observed values, False for missing
    
    # Early exit if no data to impute
    if mask.sum() == 0:
        print("All values missing. Skipping.")
        return df
    
    # Store original means and stds for later denormalization
    feature_means = np.zeros(len(input_columns))
    feature_stds = np.zeros(len(input_columns))
    
    for i in range(data.shape[1]):
        valid_idx = np.where(mask[:, i] == 1)[0]
        if len(valid_idx) > 0:
            feature_means[i] = np.mean(data[valid_idx, i])
            feature_stds[i] = np.std(data[valid_idx, i])
            if feature_stds[i] == 0:
                feature_stds[i] = 1.0
    
    # Prepare dataloader with normalized data
    dataloader = prepare_dataloader(data, mask.astype(np.float32))
    
    # Initialize and train model
    model = CorruptionAwareVAE(input_dim=len(input_columns), hidden_dim=32, latent_dim=8)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    try:
        train_vae(model, dataloader, optimizer, epochs=epochs, device=device)
    except Exception as e:
        print(f"Error during training: {e}")
        return df
    
    # Create normalized input for imputation
    normalized_data = np.copy(data)
    for i in range(data.shape[1]):
        valid_idx = np.where(mask[:, i] == 1)[0]
        if len(valid_idx) > 0:
            normalized_data[valid_idx, i] = (data[valid_idx, i] - feature_means[i]) / feature_stds[i]
    
    # Replace NaNs with zeros for model input
    model_input = np.nan_to_num(normalized_data, nan=0.0)
    
    # Perform imputation
    model.eval()
    with torch.no_grad():
        # Prepare inputs
        x_tensor = torch.tensor(model_input, dtype=torch.float32).to(device)
        mask_tensor = torch.tensor(mask.astype(np.float32), dtype=torch.float32).to(device)
        
        # Get predictions
        x_hat, _, _ = model(x_tensor, mask_tensor)
        
        # Move back to CPU and convert to numpy
        x_hat = x_hat.cpu().numpy()
        
        # Create imputed dataset (normalized scale)
        imputed_normalized = np.copy(normalized_data)
        missing_mask = ~mask
        imputed_normalized[missing_mask] = x_hat[missing_mask]
        
        # Denormalize back to original scale
        imputed_original = np.copy(imputed_normalized)
        for i in range(data.shape[1]):
            imputed_original[:, i] = imputed_normalized[:, i] * feature_stds[i] + feature_means[i]
        
        # Update missing values directly in the original DataFrame
        for i, col in enumerate(input_columns):
            missing_mask_col = missing_mask[:, i]
            if missing_mask_col.any():
                # Update only the missing values in the original column
                df.loc[df.index[missing_mask_col], col] = imputed_original[missing_mask_col, i]
        
        # Generate imputation report
        if 'subject_id' in df.columns:
            subject_id = df['subject_id'].iloc[0]
            print(f"Subject {subject_id} - Imputation report (vae):")
        else:
            print(f"Imputation report (vae):")
            
        for i, col in enumerate(input_columns):
            # Original missing count
            original_missing = np.sum(np.isnan(data[:, i]))
            
            # Current missing count (should be 0)
            current_missing = np.sum(pd.isna(df[col]))
            
            # Report
            print(f"  {col}: {original_missing} missing → {current_missing} missing (imputed {original_missing - current_missing})")
            
        # Add a summary row to clearly show success
        total_original_missing = np.sum(np.isnan(data))
        total_current_missing = sum([np.sum(pd.isna(df[col])) for col in input_columns])
        total_imputed = total_original_missing - total_current_missing
        
        if total_original_missing > 0:
            print(f"  Summary: Imputed {total_imputed}/{total_original_missing} missing values ({total_imputed/total_original_missing*100:.1f}%)")
        else:
            print("  Summary: No missing values to impute")
    
    return df