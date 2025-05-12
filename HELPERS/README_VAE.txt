# Missing Data Imputation using Variational Autoencoders (VAEs)

## Overview

This repository contains a specialized implementation of a Variational Autoencoder (VAE) designed to handle missing data in time series, 
specifically for health sensor data like gait speed and daily step counts. This README explains the approach in simple terms for those unfamiliar with VAEs.

## What is a VAE?

A Variational Autoencoder is a type of neural network designed to learn compact representations of data. It consists of two main parts:

1. **Encoder**: Transforms the input data into a compressed representation
2. **Decoder**: Reconstructs the original data from the compressed representation

Unlike regular autoencoders, VAEs add randomness to the compressed representation, making them generative models that can create new, realistic data samples.

## How Our VAE Handles Missing Data

Traditional approaches to missing data often use simple methods like mean imputation or last observation carried forward. Our approach uses a specialized VAE that:

1. Treats missing data as a "corruption process" where some values are observed and others are hidden
2. Uses a "corruption-aware" model that knows which values are missing and which are observed
3. Leverages patterns in the observed data to reconstruct missing values

### Key Components

- **Missingness Mask**: A binary indicator (1 = observed, 0 = missing) that tells the model which values are available
- **Corruption-Aware Encoder**: Takes both the data and the mask as input, so it knows which values are missing
- **Conditional Decoder**: Also has access to the mask, allowing for better imputation of missing values

## The Imputation Process

Our implementation follows these steps:

1. **Data Preparation**:
   - Extract data columns with missing values
   - Create a missingness mask (True for observed, False for missing)
   - Normalize the data to improve training stability

2. **Model Training**:
   - The VAE learns patterns from the observed data
   - The model conditions on both the data and the missingness mask
   - Training optimizes a special loss function designed for missing data

3. **Imputation**:
   - The trained model predicts values for the missing elements
   - Predicted values are denormalized back to the original scale
   - Original data is updated with the imputed values

4. **Reporting**:
   - A detailed report shows how many values were successfully imputed
   - Diagnostic plots can visualize the imputation quality

## How This Is Different From Standard Approaches

Our VAE approach offers several advantages over traditional methods:

1. **Learns Complex Patterns**: VAEs can capture non-linear relationships in your data
2. **Handles Different Missing Data Types**: Works for both Missing Completely At Random (MCAR) and Missing Not At Random (MNAR) data
3. **Probabilistic Framework**: The model learns a distribution over possible values rather than just point estimates
4. **Maintains Data Distributions**: Preserves the statistical properties of the original data

## How VAE Handles Uncertainty in Imputation

When dealing with missing data, there are often multiple plausible values that could fill each gap. Here's how our VAE approach handles this uncertainty:

1. **Probabilistic Model**: The VAE learns a probability distribution over possible values for each missing point.

2. **Expected Value Imputation**: When imputing, the model uses the mean (expected value) of the learned distribution for each missing value - essentially the "best guess" based on patterns in your data.

3. **Deterministic Output**: Once trained, the VAE produces consistent imputation results for the same input data, using the expected value from the distribution rather than randomly sampling possible values.

4. **No Selection Between Alternatives**: Unlike some multiple imputation methods that generate several datasets, our implementation provides a single imputation using the most likely value according to the model.

This approach balances the need to acknowledge uncertainty (by learning a distribution) with the practical need for a single complete dataset for downstream analysis.

## Using The Code

The main function `impute_subject_data()` takes a DataFrame with missing values and returns the same DataFrame with imputed values:

```python
# Example usage
imputed_df = impute_subject_data(
    df=your_dataframe, 
    input_columns=['gait_speed', 'daily_steps'], 
    epochs=30
)
```

## Theory Behind the Approach

This implementation is based on the paper "VAEs in the Presence of Missing Data" by Collier et al., which introduces a novel approach to handling missing data with VAEs. The key insight is modeling missing data as a corruption process and using a conditional VAE where both the encoder and decoder have access to information about which values are missing.

## References

- Collier, M., Nazabal, A., & Williams, C. K. I. (2021). VAEs in the Presence of Missing Data.
- Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes.

checklist = """
# Corruption-Aware VAE Imputation - Paper Fidelity Checklist
Based on: "VAEs in the Presence of Missing Data" by Collier et al.

✅ = Implemented in current pipeline
⚠️ = Partial or optional
❌ = Not yet implemented

------------------------------------------
✅ Core Modeling
------------------------------------------
✅ Encoder input includes missingness mask (x_tilde + mask)
✅ Decoder input includes missingness mask (z + mask)
✅ VAE architecture using reparameterization trick
✅ KL divergence regularized loss (β-weighted)
✅ MSE reconstruction loss focused only on missing values
✅ Gradient clipping, layer normalization for stability
✅ Multiple architectural variants (ZI, Encoder Mask, Encoder+Decoder Mask)

------------------------------------------
✅ Training & Evaluation
------------------------------------------
✅ Subject-wise imputation using per-feature normalization
✅ Logs ELBO, KL, MSE_obs, MSE_miss per epoch
✅ Output is used directly in downstream LSTM modeling
✅ Configurable epochs, batch size, and model variant
✅ Clean handling of tabular clinical time-series data

------------------------------------------
⚠️ Optional Fidelity Enhancements
------------------------------------------
⚠️ Monte Carlo estimate of marginal likelihood (not essential for tabular)
⚠️ Multiple samples of latent z for stochastic imputation
⚠️ Best model checkpointing via ELBO tracking
⚠️ Logistic/KNN classifier evaluation on latent z embeddings

------------------------------------------
❌ Not Implemented
------------------------------------------
❌ Image-based convolutional encoder/decoder (not applicable)
❌ Logistic mixture likelihood (you use MSE/Gaussian for tabular)

------------------------------------------
🧠 Summary
This implementation captures the essential structure and intent of the paper's corruption-aware VAE, adapted effectively for longitudinal clinical time series. It is suitable for imputation, analysis, and downstream modeling, with optional extensions available for deeper evaluation.

"""

with open("/mnt/data/vae_paper_fidelity_checklist.txt", "w") as f:
    f.write(checklist.strip())

"/mnt/data/vae_paper_fidelity_checklist.txt"