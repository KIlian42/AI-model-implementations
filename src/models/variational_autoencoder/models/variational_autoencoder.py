import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.dropout = nn.Dropout(0.2)
        
        self.encoder_fc1 = nn.Linear(28 * 28, 128)
        self.encoder_bn1 = nn.BatchNorm1d(128)
        self.encoder_fc2 = nn.Linear(128, 64)
        self.encoder_bn2 = nn.BatchNorm1d(64)

        self.fc_mean = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        self.decoder_fc1 = nn.Linear(latent_dim, 64)
        self.decoder_bn1 = nn.BatchNorm1d(64)
        self.decoder_fc2 = nn.Linear(64, 128)
        self.decoder_bn2 = nn.BatchNorm1d(128)
        self.decoder_fc3 = nn.Linear(128, 28 * 28)
    
    def encode(self, x):
        x = x.view(x.size(0), -1) # [Batch, 1, 28, 28] → [Batch, 784]
        h = self.encoder_fc1(x)
        h = self.encoder_bn1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.encoder_fc2(h)
        h = self.encoder_bn2(h)
        h = F.relu(h)
        h = self.dropout(h)
        mu = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # Sample
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc1(z)
        h = self.decoder_bn1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.decoder_fc2(h)
        h = self.decoder_bn2(h)
        h = F.relu(h)
        h = self.dropout(h)
        out = self.decoder_fc3(h)
        out = torch.sigmoid(out)  # Normalize → [0, 1]
        out = out.view(-1, 1, 28, 28)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar

    def get_latent(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z
