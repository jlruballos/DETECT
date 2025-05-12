import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from corruption_aware_vae import get_model_variant

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def prepare_dataloader(data, masks, batch_size=64):
    scaler = StandardScaler()
    clean_data = np.copy(data)

    for i in range(data.shape[1]):
        valid_idx = np.where(masks[:, i] == 1)[0]
        if len(valid_idx) > 0:
            mean = np.mean(data[valid_idx, i])
            std = np.std(data[valid_idx, i]) or 1.0
            clean_data[:, i] = (data[:, i] - mean) / std

    clean_data = np.nan_to_num(clean_data, nan=0.0)
    x = torch.tensor(clean_data, dtype=torch.float32)
    mask = torch.tensor(masks, dtype=torch.float32)
    x_tilde = x.clone()
    dataset = TensorDataset(x, x_tilde, mask)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), scaler

def impute_subject_data(df, input_columns, epochs=30, variant="Encoder + Decoder Mask"):
    data = df[input_columns].values.astype(np.float32)
    mask = ~np.isnan(data)

    if mask.sum() == 0:
        print("All values missing. Skipping.")
        return df

    dataloader, _ = prepare_dataloader(data, mask.astype(np.float32))
    model = get_model_variant(variant, input_dim=len(input_columns)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"Training VAE variant: {variant}")
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

    # Impute after training
    model.eval()
    with torch.no_grad():
        normalized_data = np.copy(data)
        means = np.zeros(data.shape[1])
        stds = np.ones(data.shape[1])
        for i in range(data.shape[1]):
            idx = np.where(mask[:, i])[0]
            if len(idx) > 0:
                means[i] = np.mean(data[idx, i])
                stds[i] = np.std(data[idx, i]) or 1.0
                normalized_data[idx, i] = (data[idx, i] - means[i]) / stds[i]
        input_tensor = torch.tensor(np.nan_to_num(normalized_data, nan=0.0), dtype=torch.float32).to(device)
        mask_tensor = torch.tensor(mask.astype(np.float32), dtype=torch.float32).to(device)
        x_hat, _, _ = model(input_tensor, mask_tensor)
        x_hat = x_hat.cpu().numpy()

        imputed = np.copy(normalized_data)
        missing_mask = ~mask
        imputed[missing_mask] = x_hat[missing_mask]
        for i in range(data.shape[1]):
            imputed[:, i] = imputed[:, i] * stds[i] + means[i]

        for i, col in enumerate(input_columns):
            df.loc[df[col].isna(), col] = imputed[:, i][np.isnan(data[:, i])]

    return df