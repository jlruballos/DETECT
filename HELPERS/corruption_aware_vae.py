import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CorruptionAwareVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16,
                 encoder_mask=True, decoder_mask=True):
        super().__init__()
        self.encoder_mask = encoder_mask
        self.decoder_mask = decoder_mask

        encoder_input_dim = input_dim * 2 if encoder_mask else input_dim
        decoder_input_dim = latent_dim + input_dim if decoder_mask else latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x_tilde, mask):
        enc_input = torch.cat([x_tilde, mask], dim=1) if self.encoder_mask else x_tilde
        h = self.encoder(enc_input)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * torch.clamp(logvar, min=-10, max=10))
        return mu + std * torch.randn_like(std)

    def decode(self, z, mask):
        dec_input = torch.cat([z, mask], dim=1) if self.decoder_mask else z
        return self.decoder(dec_input)

    def forward(self, x_tilde, mask):
        mu, logvar = self.encode(x_tilde, mask)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, mask)
        return x_hat, mu, logvar

    def loss_function(self, x, x_hat, mu, logvar, mask):
        missing_mask = 1.0 - mask
        observed_mask = mask

        recon_loss = torch.sum(((x_hat - x) * missing_mask) ** 2) / (missing_mask.sum() + 1e-8)
        mse_obs = torch.sum(((x_hat - x) * observed_mask) ** 2) / (observed_mask.sum() + 1e-8)
        mse_miss = torch.sum(((x_hat - x) * missing_mask) ** 2) / (missing_mask.sum() + 1e-8)

        kl_loss = -0.5 * torch.mean(1 + torch.clamp(logvar, -10, 10) - mu.pow(2) - torch.exp(torch.clamp(logvar, -10, 10)))
        beta = 0.1
        total_loss = recon_loss + beta * kl_loss

        return total_loss, recon_loss.item(), kl_loss.item(), mse_obs.item(), mse_miss.item()

def get_model_variant(name, input_dim):
    name = name.strip().lower()
    if name in ["zero imputation", "zero"]:
        return CorruptionAwareVAE(input_dim, encoder_mask=False, decoder_mask=False)
    elif name in ["encoder mask", "zero imputation + encoder mask"]:
        return CorruptionAwareVAE(input_dim, encoder_mask=True, decoder_mask=False)
    elif name in ["encoder + decoder mask", "zero imputation + encoder decoder mask"]:
        return CorruptionAwareVAE(input_dim, encoder_mask=True, decoder_mask=True)
    else:
        raise ValueError(f"Unknown method: {name}")