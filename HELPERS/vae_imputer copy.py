#!/usr/bin/env python3
"""
Corruption-Aware Variational Autoencoder (VAE) for Imputation

This model implements a VAE that takes in corrupted data (with missing values filled)
and a binary mask indicating which values were originally observed. Both the encoder
and decoder are conditioned on this mask.

This VAE learns to reconstruct missing sensor data using both the corrupted input 
(with placeholder values) and a binary mask that indicates which values are missing. 
The mask is passed to both the encoder and decoder, enabling the model to learn the pattern of missingness. 
During training, reconstruction loss is calculated only on missing values, encouraging the model to learn 
accurate imputations without being biased by observed data.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CorruptionAwareVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super(CorruptionAwareVAE, self).__init__()
		
		# Encoder
        self.encoder = nn.Sequential(
			nn.Linear(input_dim * 2, hidden_dim),  # Input is concatenated data and mask
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
		)
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # Mean of the latent space
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Log variance of the latent space
        
        #decoder: input = [z, mask]
        self.decoder = nn.Sequential(
			nn.Linear(latent_dim + input_dim, hidden_dim),  # Input is concatenated latent vector and mask
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, input_dim)
		)
        
    def encode(self, x_tilde, mask):
        input_combined = torch.cat([x_tilde, mask], dim=1)  # Concatenate data and mask
        h = self.encoder(input_combined)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z, mask):
        input_combined = torch.cat([z, mask], dim=1)  # Concatenate latent vector and mask
        return self.decoder(input_combined)
        
    def forward(self, x_tilde, mask):
        mu, logvar = self.encode(x_tilde, mask)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, mask)
        print(f"x_tilde shape: {x_tilde.shape}, x_hat shape: {x_hat.shape}")
        return x_hat, mu, logvar
        
    def loss_function(self, x, x_hat, mu, logvar, mask):
		#compute MSE only on originally missing values (mask == 0)
        # recon_loss = F.mse_loss(x_hat * (1 - mask), x * (1 - mask), reduction='sum')
        # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # return recon_loss + kl_loss, recon_loss, kl_loss
        epsilon = 1e-8
    
		# Compute MSE only on originally missing values (mask == 0)
		# Using mean instead of sum can help with stability
        missing_mask = 1 - mask
        n_missing = torch.sum(missing_mask) + epsilon
		
		# Only compute loss on missing values
        recon_loss = torch.sum(((x_hat * missing_mask - x * missing_mask) ** 2)) / n_missing
		
		# More stable KL divergence calculation
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
		
		# Balance the two loss components
        beta = 0.1  # Reduce the weight of KL loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
        
def train_vae(model, dataloader, optimizer, epochs=50, device=device):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, x_tilde, mask in dataloader:
            x = x.to(device)
            x_tilde = x_tilde.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            x_hat, mu, logvar = model(x_tilde, mask)
            loss, recon_loss, kl_loss = model.loss_function(x, x_hat, mu, logvar, mask)
            loss.backward()
            
             # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.2f}")
        
def prepare_dataloader(data, masks, batch_size=64):
    #Assumes data and masks are numpy arrays with shape (N, D)
    x = torch.tensor(data, dtype=torch.float32)
    mask = torch.tensor(masks, dtype=torch.float32)
    x_tilde = x.clone()
    x_tilde[mask == 0] = 0  # Replace missing values with 0 or any other placeholder
    dataset = TensorDataset(x, x_tilde, mask)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True) # Create DataLoader for training

def impute_subject_data(df, input_columns, epochs=30, device=device):
    data = df[input_columns].values.astype(np.float32)
    mask = ~np.isnan(data)

    if mask.sum() == 0:
        print("All values missing. Skipping.")
        return df

    # Normalize the data to help with training stability
    data_mean = np.nanmean(data, axis=0)
    data_std = np.nanstd(data, axis=0)
    data_std[data_std == 0] = 1.0  # Avoid division by zero
    
    # Create normalized version of the data
    data_normalized = np.copy(data)
    for i in range(data.shape[1]):
        data_normalized[:, i] = (data[:, i] - data_mean[i]) / data_std[i]
        # Only normalize non-NaN values
        data_normalized[np.isnan(data[:, i]), i] = np.nan

    # Prepare masked input (with zeros replacing NaNs)
    masked_input = np.copy(data_normalized)
    masked_input[~mask] = 0
    
    # Prepare dataloader
    dataloader = prepare_dataloader(data_normalized, mask.astype(np.float32))

    # Create and train model
    model = CorruptionAwareVAE(input_dim=len(input_columns), hidden_dim=32, latent_dim=8)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Reduced learning rate
    
    # Train the model
    train_vae(model, dataloader, optimizer, epochs=epochs, device=device)

    # Impute missing values
    model.eval()
    with torch.no_grad():
        # Convert inputs to tensors and move to device
        full_input = torch.tensor(masked_input, dtype=torch.float32).to(device)
        full_mask = torch.tensor(mask.astype(np.float32), dtype=torch.float32).to(device)
        
        # Get model predictions
        x_hat, _, _ = model(full_input, full_mask)
        
        # Convert everything to tensors for consistent operations
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        
        # Combine observed and imputed values (all as tensors)
        imputed = mask_tensor.to(device) * full_input + (1 - mask_tensor.to(device)) * x_hat
        
        # Convert back to CPU and numpy for dataframe insertion
        imputed_np = imputed.cpu().numpy()
        
        # Denormalize the imputed values
        imputed_denorm = np.copy(imputed_np)
        for i in range(data.shape[1]):
            imputed_denorm[:, i] = imputed_np[:, i] * data_std[i] + data_mean[i]
        
        # Add imputed values to dataframe
        for i, col in enumerate(input_columns):
            df[col + '_vae'] = imputed_denorm[:, i]
            
            # Copy original values where they exist
            df.loc[~np.isnan(data[:, i]), col + '_vae'] = df.loc[~np.isnan(data[:, i]), col]
            
    return df