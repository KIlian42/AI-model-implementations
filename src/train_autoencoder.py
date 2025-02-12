import torch
from torch import nn
import torch.optim as optim
from models.variational_autoencoder.models.autoencoder import Autoencoder
from util.print_model_architecture import print_layer_shapes
from util.mnist_dataloader import load_mnist_dataset
from tqdm import tqdm

NUM_EPOCHS = 10
PATIENCE = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def evaluate_autoencoder(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            imgs = batch["image"].to(device)
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            total_loss += loss.item() * imgs.size(0)
    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss

if __name__ == "__main__":
    train_loader, test_loader = load_mnist_dataset()
    autoencoder_model = Autoencoder().to(device)
    # dummy_input = torch.randn(1, 1, 28, 28) # (Batch, Channels, Width, Height)
    # print_layer_shapes(model, dummy_input)

    # ---------------------------------
    # Training Autoencoder (Rekonstruktion, MSELoss)
    # ---------------------------------
    print("Training Autoencoder")
    criterion_ae = nn.MSELoss()
    optimizer_ae = optim.Adam(autoencoder_model.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(NUM_EPOCHS):
        autoencoder_model.train()
        progress_bar = tqdm(train_loader, desc=f"Autoencoder Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")
        for batch in progress_bar:
            imgs = batch["image"].to(device)
            outputs = autoencoder_model(imgs)
            loss = criterion_ae(outputs, imgs)

            optimizer_ae.zero_grad()
            loss.backward()
            optimizer_ae.step()

            progress_bar.set_postfix(loss=loss.item())

        val_loss = evaluate_autoencoder(autoencoder_model, test_loader, criterion_ae, device)
        print(f"Autoencoder Epoch {epoch+1} Validation Loss: {val_loss:.4f}")

        # Early Stopping Logik
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Hier kann man den Modelldump speichern, z. B. torch.save(...)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping for Autoencoder")
                break