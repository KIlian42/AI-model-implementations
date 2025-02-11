import torch
import torch.nn as nn

# ---------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# ---------------------------------------------------------------
# Residual CNN original
# ---------------------------------------------------------------
class ResidualCNNC(nn.Module):
    def __init__(self):
        super(ResidualCNNC, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = ResidualBlock(16, 16, stride=1)
        self.layer2 = ResidualBlock(16, 32, stride=2)
        self.layer3 = ResidualBlock(32, 64, stride=2)
        self.layer4 = ResidualBlock(64, 64, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ---------------------------------------------------------------
# Residual Block für vollständig verbundene (FC) Schichten
# ---------------------------------------------------------------
class ResidualBlockFC(nn.Module):
    def __init__(self, features: int, hidden_features: int = None):
        """
        Ein Residual Block für Fully Connected Netzwerke.
        """
        super(ResidualBlockFC, self).__init__()
        if hidden_features is None:
            hidden_features = features
        self.fc1 = nn.Linear(features, hidden_features)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_features, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += identity
        out = self.relu(out)
        return out


# ---------------------------------------------------------------
# Klassifizierer für den latenten Raum (64-dim Vektor)
# ---------------------------------------------------------------
class LatentClassifier(nn.Module):
    def __init__(self, in_features: int = 64, num_classes: int = 10, num_blocks: int = 4):
        super(LatentClassifier, self).__init__()
        self.initial_fc = nn.Linear(in_features, in_features)
        # Erzeuge 'num_blocks' Residual Blöcke
        self.res_blocks = nn.Sequential(*[ResidualBlockFC(in_features) for _ in range(num_blocks)])
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_fc(x)
        x = self.res_blocks(x)
        x = self.fc(x)
        return x
