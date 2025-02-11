import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # Normiert die Ausgaben auf den Bereich [0, 1]
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1) # [Batch, 1, 28, 28] → [Batch, 784]
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(x.size(0), 1, 28, 28) # [Batch, 784] → [Batch, 1, 28, 28]
        return decoded
    
    def get_latent(self, x):
        x = x.view(x.size(0), -1)
        latent = self.encoder(x)
        return latent
