import torch
from torch import nn
import torch.optim as optim
from models.convolutional_network.models.residual_cnn import ResidualCNN
from util.print_model_architecture import print_layer_shapes
from util.mnist_dataloader import load_mnist_dataset

from tqdm import tqdm

NUM_EPOCHS = 10
PATIENCE = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def evaluate_residualcnn(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy

if __name__ == "__main__":
    train_loader, test_loader = load_mnist_dataset()
    residualcnn_model = ResidualCNN()

    dummy_input = torch.randn(1, 1, 28, 28) # (Batch, Channels, Width, Height)
    print_layer_shapes(residualcnn_model, dummy_input)

    # ---------------------------------
    # Training ResidualCNN (Klassifikation, CrossEntropyLoss)
    # ---------------------------------
    print("\nTraining ResidualCNN")
    criterion_cls = nn.CrossEntropyLoss()
    optimizer_res = optim.Adam(residualcnn_model.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(NUM_EPOCHS):
        residualcnn_model.train()
        progress_bar = tqdm(train_loader, desc=f"ResidualCNN Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")
        for batch in progress_bar:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = residualcnn_model(imgs)
            loss = criterion_cls(outputs, labels)

            optimizer_res.zero_grad()
            loss.backward()
            optimizer_res.step()

            progress_bar.set_postfix(loss=loss.item())

        val_loss, val_acc = evaluate_residualcnn(residualcnn_model, test_loader, criterion_cls, device)
        print(f"ResidualCNN Epoch {epoch+1} Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping for ResidualCNN")
                break
