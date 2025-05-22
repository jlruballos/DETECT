import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from corruption_aware_vae import get_model_variant

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def prepare_global_dataloader(all_subject_data, all_subject_masks, batch_size=64):
    scaler = StandardScaler()
    full_data = np.vstack(all_subject_data)
    full_mask = np.vstack(all_subject_masks)

    means = []
    stds = []
    clean_data = np.copy(full_data)

    for i in range(full_data.shape[1]):
        observed_values = full_data[full_mask[:, i] == 1, i]
        mean = np.mean(observed_values)
        std = np.std(observed_values) or 1.0
        means.append(mean)
        stds.append(std)
        clean_data[:, i] = (full_data[:, i] - mean) / std

    clean_data = np.nan_to_num(clean_data, nan=0.0)

    scaler.mean_ = np.array(means)
    scaler.scale_ = np.array(stds)

    x = torch.tensor(clean_data, dtype=torch.float32)
    mask = torch.tensor(full_mask, dtype=torch.float32)
    x_tilde = x.clone()
    dataset = TensorDataset(x, x_tilde, mask)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), scaler

def train_global_vae(dataloader, input_dim, variant='Encoder + Decoder Mask', epochs=30):
    model = get_model_variant(variant, input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"Training GLOBAL VAE variant: {variant}")
    model.train()
    for epoch in range(epochs):
        total_loss, total_kl, total_recon, total_mse_obs, total_mse_miss = 0, 0, 0, 0, 0
        batches = 0
        for x, x_tilde, m in dataloader:
            x, x_tilde, m = x.to(device), x_tilde.to(device), m.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = model(x_tilde, m)
            loss, recon, kl, mse_obs, mse_miss = model.loss_function(x, x_hat, mu, logvar, m)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon += recon
            total_kl += kl
            total_mse_obs += mse_obs
            total_mse_miss += mse_miss
            batches += 1
        print(f"Epoch {epoch+1}: ELBO={-total_loss/batches:.4f}, KL={total_kl/batches:.4f}, Recon={total_recon/batches:.4f}, MSE_obs={total_mse_obs/batches:.4f}, MSE_miss={total_mse_miss/batches:.4f}")
    return model

def impute_with_trained_vae(df, input_columns, model, scaler):
    data = df[input_columns].values.astype(np.float32)
    mask = ~np.isnan(data)

    # Normalize with the same scaler
    norm_data = scaler.transform(np.nan_to_num(data, nan=0.0))
    input_tensor = torch.tensor(norm_data, dtype=torch.float32).to(device)
    mask_tensor = torch.tensor(mask.astype(np.float32), dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        x_hat, _, _ = model(input_tensor, mask_tensor)
        x_hat = x_hat.cpu().numpy()

    imputed = np.copy(data)
    imputed[~mask] = x_hat[~mask]

    for i, col in enumerate(input_columns):
        df.loc[df[col].isna(), col] = imputed[:, i][np.isnan(data[:, i])]

    return df