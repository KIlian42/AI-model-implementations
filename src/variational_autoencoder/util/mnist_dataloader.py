from datasets import load_dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_mnist_dataset():
    mnist_dataset = load_dataset("mnist")
    transform = transforms.ToTensor()

    def _transform_example(example):
        example["image"] = transform(example["image"])
        return example
    
    mnist_dataset = mnist_dataset.map(_transform_example)
    mnist_dataset.set_format(type="torch", columns=["image", "label"])

    train_dataset = mnist_dataset["train"]
    test_dataset = mnist_dataset["test"]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_loader, test_loader


