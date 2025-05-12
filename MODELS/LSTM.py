#!/usr/bin/env python3
"""
Train LSTM on DETECT Sequences with Configurable Cross-Validation

- Loads .npz file with (X, y, subid)
- Uses cross-validation strategy (e.g., groupkfold)
- Trains and evaluates a simple LSTM classifier
- Logs fold-wise metrics (accuracy, F1, AUC) using Weights & Biases
- Saves best model per fold
- Tracks GPU or CPU usage
- Logs GPU memory usage if available
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
# -------- CONFIG --------
USE_WSL = True  # Set to True if running inside WSL (Linux on Windows)
# Define base paths depending on environment
if USE_WSL:
    base_path = '/mnt/d/DETECT'
else:
    base_path = r'D:\DETECT'
    
# Add helpers directory to system path
sys.path.append(os.path.join(base_path, 'HELPERS'))
from helpers import get_cv_splits
import wandb  # Weights & Biases for logging

# ---- Define the LSTM Model ----
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 
                            hidden_size, 
                            num_layers = 2,
                            batch_first=True, 
                            dropout=dropout,
                            bidirectional=True)
        
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out  # Note: removed sigmoid

# ---- Train one fold of CV ----
def train_one_fold(model, train_loader, val_loader, epochs, device, fold_num):
    # Calculate positive class weight for imbalanced classification
    all_labels = torch.cat([y for _, y in train_loader])
    pos_weight = (all_labels == 0).sum() / (all_labels == 1).sum()
    pos_weight = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_auc = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).squeeze()
            loss = criterion(logits, yb.float())
            loss.backward()
            optimizer.step()

        # ---- Evaluation ----
        model.eval()
        val_logits, val_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb).squeeze().cpu().numpy()
                val_logits.extend(logits)
                val_labels.extend(yb.numpy())

        val_probs = torch.sigmoid(torch.tensor(val_logits)).numpy()
        val_preds_binary = (val_probs > 0.5).astype(int)
        acc = accuracy_score(val_labels, val_preds_binary)
        f1 = f1_score(val_labels, val_preds_binary)
        auc = roc_auc_score(val_labels, val_probs)
        precision = precision_score(val_labels, val_preds_binary)
        recall = recall_score(val_labels, val_preds_binary)

        log_dict = {
            f"Fold{fold_num}/Val_Accuracy": acc,
            f"Fold{fold_num}/Val_F1": f1,
            f"Fold{fold_num}/Val_AUC": auc,
            f"Fold{fold_num}/Val_Precision": precision,
            f"Fold{fold_num}/Val_Recall": recall,
            f"Fold{fold_num}/Epoch": epoch + 1
        }

        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1024**2
            log_dict[f"Fold{fold_num}/GPU_Memory_MB"] = allocated

        # Log to Weights & Biases
        wandb.log(log_dict)

        if auc > best_val_auc:
            best_val_auc = auc
            best_model_state = model.state_dict()

        print(f"  Epoch {epoch+1}: Val Acc={acc:.3f}, F1={f1:.3f}, AUC={auc:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")

    return best_model_state, best_val_auc

# ---- Main Training Loop ----
def run_lstm_cv(npz_path, cv_method='groupkfold', n_splits=5,
                hidden_size=64, batch_size=32, epochs=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(project="detect-lstm", config={
        "cv_method": cv_method,
        "hidden_size": hidden_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "data_path": npz_path,
        "device": str(device)
    })

    print(f"Loading sequences from: {npz_path}")
    data = np.load(npz_path)
    X, y, subid = data['X'], data['y'], data['subid']

    splits = get_cv_splits(X, y, subid, method=cv_method, n_splits=n_splits)

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")

        X_train = torch.tensor(X[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y[train_idx], dtype=torch.float32)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y[val_idx], dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

        model = LSTMClassifier(input_size=X.shape[2], hidden_size=hidden_size).to(device)
        best_model_state, best_auc = train_one_fold(model, train_loader, val_loader, epochs, device, fold + 1)

        out_dir = os.path.join(os.path.dirname(npz_path), 'models')
        os.makedirs(out_dir, exist_ok=True)
        ckpt_path = os.path.join(out_dir, f'lstm_fold{fold+1}.pt')
        torch.save(best_model_state, ckpt_path)
        print(f"Saved best model to {ckpt_path}")

        fold_metrics.append(best_auc)

    print("\n==== Cross-Validation Summary ====")
    print(f"AUCs: {fold_metrics}")
    print(f"Mean AUC: {np.mean(fold_metrics):.4f}")
    wandb.finish()

# ---- Entry Point ----
if __name__ == '__main__':
    run_lstm_cv(
        npz_path='/mnt/d/DETECT/OUTPUT/generate_lstm_sequences/lstm_sequences_vae.npz',
        cv_method='groupkfold',
        n_splits=5,
        hidden_size=64,
        batch_size=32,
        epochs=10
    )
