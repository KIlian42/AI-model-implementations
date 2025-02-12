import torch
from torch import nn
import torch.optim as optim
from models.variational_autoencoder.models.autoencoder import Autoencoder
from util.print_model_architecture import print_layer_shapes
from util.mnist_dataloader import load_mnist_dataset
from tqdm import tqdm

NUM_EPOCHS = 5 

if __name__ == "__main__":
    train_loader, test_loader = load_mnist_dataset()
    model = Autoencoder()
    # dummy_input = torch.randn(1, 1, 28, 28) # (Batch, Channels, Width, Height)
    # print_layer_shapes(model, dummy_input)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(NUM_EPOCHS):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")
        for batch_idx, batch in enumerate(progress_bar):
            imgs = batch["image"]
            
            # Forward Pass
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix(loss=loss.item())