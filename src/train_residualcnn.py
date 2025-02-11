import torch
from torch import nn
import torch.optim as optim
from models.convolutional_network.models.residual_cnn import ResidualCNN
from util.print_model_architecture import print_layer_shapes
from util.mnist_dataloader import load_mnist_dataset

from tqdm import tqdm

if __name__ == "__main__":
    train_loader, test_loader = load_mnist_dataset()
    model = ResidualCNN()

    dummy_input = torch.randn(1, 1, 28, 28) # (Batch, Channels, Width, Height)
    print_layer_shapes(model, dummy_input)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    NUM_EPOCHS = 5

    for epoch in range(NUM_EPOCHS):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")
        for batch_idx, batch in enumerate(progress_bar):
            imgs = batch["image"]    # Shape: [Batch, 1, 28, 28]
            labels = batch["label"]  # Shape: [Batch]
            
            # Forward Pass
            outputs = model(imgs)    
            loss = criterion(outputs, labels)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix(loss=loss.item())
