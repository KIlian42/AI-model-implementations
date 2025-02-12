import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from models.variational_autoencoder.models.variational_autoencoder import VariationalAutoencoder
from util.print_model_architecture import print_layer_shapes
from util.mnist_dataloader import load_mnist_dataset

from tqdm import tqdm

NUM_EPOCHS = 10
PATIENCE = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Lossfunktion für den VAE: Rekonstruktionsverlust + KL-Divergenz
def vae_loss(recon_x, x, mu, logvar):
    # Binary Cross-Entropy (BCE) als Rekonstruktionsverlust
    # Wir flachen beide Tensoren zu [Batch, 784] ab.
    BCE = F.binary_cross_entropy(recon_x.view(-1, 28*28), x.view(-1, 28*28), reduction='sum')
    # KL-Divergenz: Vergleicht die latente Verteilung mit einer Standardnormalverteilung
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def evaluate_varationalautoencoder(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            imgs = batch["image"].to(device)
            # Bei einem VAE liefert der Forward-Pass (rekonstruierte Bilder, mu, logvar)
            recon_x, mu, logvar = model(imgs)
            loss = vae_loss(recon_x, imgs, mu, logvar)
            # Da wir "sum" als Reduction verwenden, summieren wir bereits über die Batch-Größe
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss

if __name__ == "__main__":
    train_loader, test_loader = load_mnist_dataset()
    variational_autoencoder_model = VariationalAutoencoder()

    dummy_input = torch.randn(1, 1, 28, 28) # (Batch, Channels, Width, Height)
    print_layer_shapes(variational_autoencoder_model, dummy_input)

    # ---------------------------------
    # Training Variational Autoencoder
    # ---------------------------------
    print("Training VAE")
    optimizer_vae = optim.Adam(variational_autoencoder_model.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    epochs_no_improve = 0


    for epoch in range(NUM_EPOCHS):
        variational_autoencoder_model.train()
        progress_bar = tqdm(train_loader, desc=f"VAE Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")
        for batch in progress_bar:
            imgs = batch["image"].to(device)
            recon_x, mu, logvar = variational_autoencoder_model(imgs)
            loss = vae_loss(recon_x, imgs, mu, logvar)
            
            optimizer_vae.zero_grad()
            loss.backward()
            optimizer_vae.step()
            
            progress_bar.set_postfix(loss=loss.item())
        
        val_loss = evaluate_varationalautoencoder(variational_autoencoder_model, test_loader, device)
        print(f"VAE Epoch {epoch+1} Validation Loss: {val_loss:.4f}")
        
        # Early Stopping Logik
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Hier könntest du das Modell abspeichern, z.B.
            # torch.save(variational_autoencoder_model.state_dict(), "best_vae_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping for VAE")
                break
