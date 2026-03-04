import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.options import load_config
from models.Nets import ResNet18
import os
import numpy as np


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, device='cuda'):
    """Evaluate classification accuracy"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            preds = output.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    config = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(config.get('seed', 42))

    # --------------------------
    # Data preparation
    # --------------------------
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(config['data_path'], 'train'),
                                         transform=transform_train)
    test_dataset = datasets.ImageFolder(root=os.path.join(config['data_path'], 'test'),
                                        transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # --------------------------
    # Model initialization
    # --------------------------
    model = ResNet18(num_classes=config['num_classes']).to(device)
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # --------------------------
    # Training loop
    # --------------------------
    best_acc = 0.0
    for epoch in range(config['rounds']):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # Evaluate
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch + 1}/{config['rounds']}, Test Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_baseline_model.pth')
            print("Best model updated!")

    print(f"\nTraining completed. Best Test Accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    main()