import torch
from torch import nn
import torch.optim as optim
from variational_autoencoder.models.autoencoder_mnist import Autoencoder
from util.print_model_architecture import print_layer_shapes
from util.mnist_dataloader import load_mnist_dataset

if __name__ == "__main__":
    train_loader, test_loader = load_mnist_dataset()
    model = Autoencoder()
    dummy_input = torch.randn(1, 1, 28, 28) # (Batch, Channels, Width, Height)
    print_layer_shapes(model, dummy_input)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    NUM_EPOCHS = 5 

    for epoch in range(NUM_EPOCHS):
        for batch_idx, batch in enumerate(train_loader):
            imgs = batch["image"]
            # Forward pass
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            # Backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")