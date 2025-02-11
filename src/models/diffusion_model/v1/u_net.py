import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# Transformationen für die Bilder
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Größe anpassen
    transforms.Grayscale(num_output_channels=3),  # Konvertiere in 3 Kanäle
    transforms.ToTensor(),  # In Tensor umwandeln
    transforms.Normalize((0.5,), (0.5,))  # Normalisierung auf [-1, 1]
])

# Dataset laden (hier MNIST)
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# Dataloader definieren
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

def add_noise(x, t, noise=None):
    """
    Fügt einem Bild Rauschen basierend auf Zeit t hinzu.
    x: Bildtensor [Batch, Channels, Height, Width]
    t: Zeitindex [Batch]
    noise: Optional, falls nicht gegeben, wird normalverteiltes Rauschen erzeugt.
    """
    if noise is None:
        noise = torch.randn_like(x)
    alphas = torch.linspace(0.0001, 0.02, t.max() + 1).to(x.device)
    alpha_t = alphas[t].view(-1, 1, 1, 1)  # Formate alpha für Multiplikation
    noisy_x = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
    return noisy_x, noise

class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),  # Eingabekanäle auf 6 ändern
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_channels, 3, padding=1),
        )
    
    def forward(self, x, t):
        # Zeitinformation als zusätzlichen Kanal hinzufügen
        time_embedding = t.view(-1, 1, 1, 1).expand_as(x)
        x = torch.cat([x, time_embedding], dim=1)  # Füge Zeitkanal zu den Bildkanälen hinzu
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

def diffusion_loss(model, x, t):
    noisy_x, true_noise = add_noise(x, t)
    predicted_noise = model(noisy_x, t)
    loss = F.mse_loss(predicted_noise, true_noise)  # MSE zwischen echtem und vorhergesagtem Rauschen
    return loss

# Beispiel-Trainingsloop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 10
for epoch in range(epochs):  # Dummy-Epochen
    epoch_loss = 0  # Loss für die aktuelle Epoche
    with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:  # Fortschrittsbalken pro Epoche
        for images, _ in pbar:  # DataLoader liefert (Bilder, Labels)
            images = images.to(device)
            t = torch.randint(0, 100, (images.size(0),), device=device)  # Zeitindizes
            loss = diffusion_loss(model, images, t)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())  # Fortschrittsbalken mit aktuellem Loss anzeigen

    # Nach jeder Epoche den durchschnittlichen Loss ausgeben
    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {epoch_loss / len(dataloader):.4f}")

def sample(model, img_shape, steps=100):
    x = torch.randn(img_shape).to(device)  # Start mit reinem Rauschen
    for t in reversed(range(steps)):
        t_tensor = torch.tensor([t] * img_shape[0], device=device)
        predicted_noise = model(x, t_tensor)
        alphas = torch.linspace(0.0001, 0.02, steps).to(x.device)
        alpha_t = alphas[t]
        x = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
    return x

# Beispiel: Erzeugung eines Bildes
generated_images = sample(model, (8, 3, 64, 64))  # Batch von 8 Bildern mit 3 Kanälen, 64x64 Pixel
plt.imshow(generated_images[0].permute(1, 2, 0).cpu().detach().numpy())
plt.show()
