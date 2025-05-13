import numpy as np

npz_path='/mnt/d/DETECT/OUTPUT/generate_lstm_sequences/lstm_sequences_vae.npz'

data = np.load(npz_path)

# Extract arrays
X = data['X']
y = data['y']
subid = data['subid']

# Inspect shapes
print(f"X shape (samples, timesteps, features): {X.shape}")
print(f"y shape (samples,): {y.shape}")
print(f"subid shape: {subid.shape}")
print(f"Number of unique subids: {len(np.unique(subid))}")

# Preview first example
print("\nFirst sequence example (X[0]):")
print(X[0])

print("\nFirst label (y[0]):", y[0])
print("First subid (subid[0]):", subid[0])