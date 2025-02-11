import torch.nn as nn
from torch import Tensor

def print_layer_shapes(model: nn.Module, input_tensor: Tensor) -> None:
    """
    Registriert Hooks in jedem Modul des Models, um während eines Forward-Passes
    die Eingabe- und Ausgabeformen auszugeben. Die Ergebnisse werden anschließend
    formatiert in einer Tabelle ausgegeben.

    Args:
        model (nn.Module): Das PyTorch-Modell, dessen Layer-Formen ausgegeben werden sollen.
        input_tensor (Tensor): Ein Dummy-Eingabetensor, der dem Modell übergeben wird.
    """
    results = []  # Liste zum Speichern der Ergebnisse als Tuple: (Layer-Name, Input Shape, Output Shape)

    def hook_fn(module: nn.Module, input, output) -> None:
        module_name = module.__class__.__name__
        in_shape = input[0].shape if isinstance(input, (tuple, list)) else input.shape
        out_shape = output.shape if not isinstance(output, (tuple, list)) else output[0].shape
        # Ergebnisse als Strings speichern
        results.append((module_name, str(in_shape), str(out_shape)))

    # Registriere Hooks für alle relevanten Module (außer Container wie nn.Sequential und das Modell selbst)
    hooks = []
    for module in model.modules():
        if not isinstance(module, nn.Sequential) and module != model:
            hooks.append(module.register_forward_hook(hook_fn))

    # Führe einen Forward-Pass durch, um die Hooks zu triggern
    model(input_tensor)

    # Entferne alle registrierten Hooks
    for hook in hooks:
        hook.remove()

    # --- Ausgabe formatieren ---

    # Definiere Header
    header = ("Layer", "Input Shape", "Output Shape")
    # Berechne die maximale Breite pro Spalte (zwischen Header und allen Ergebnissen)
    col_widths = [
        max(len(header[0]), max((len(row[0]) for row in results), default=0)),
        max(len(header[1]), max((len(row[1]) for row in results), default=0)),
        max(len(header[2]), max((len(row[2]) for row in results), default=0))
    ]
    # Gesamtbreite der Tabelle (zusätzliche Leerzeichen und Spaltentrennung berücksichtigen)
    total_width = sum(col_widths) + 8

    # Dekorierter Header der Tabelle
    print("=" * total_width)
    print("Layer Output Overview".center(total_width))
    print("=" * total_width)
    # Headerzeile
    print(f"{header[0]:<{col_widths[0]}}    {header[1]:<{col_widths[1]}}    {header[2]:<{col_widths[2]}}")
    print("-" * total_width)
    # Ausgabe jeder Zeile
    for layer_name, in_shape, out_shape in results:
        print(f"{layer_name:<{col_widths[0]}}    {in_shape:<{col_widths[1]}}    {out_shape:<{col_widths[2]}}")
    print("=" * total_width)


# --- Beispiel: Anwendung der Funktion ---
# if __name__ == "__main__":
#     class SimpleCNN(nn.Module):
#         def __init__(self):
#             super(SimpleCNN, self).__init__()
#             self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
#             self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
#             self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
#             self.fc    = nn.Linear(32 * 7 * 7, 10)

#         def forward(self, x: Tensor) -> Tensor:
#             x = self.pool(torch.relu(self.conv1(x)))
#             x = self.pool(torch.relu(self.conv2(x)))
#             x = x.view(x.size(0), -1)  # Flatten
#             x = self.fc(x)
#             return x

#     # Modell instanziieren
#     model = SimpleCNN()
#     # Dummy-Eingabetensor (Batchgröße 1, 1 Kanal, 28x28 Pixel)
#     dummy_input = torch.randn(1, 1, 28, 28)
    
#     # Ausgabe der Layer-Formen
#     print_layer_shapes(model, dummy_input)
