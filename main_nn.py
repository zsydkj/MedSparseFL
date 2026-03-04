import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

from utils.options import load_config
from models.Nets import ResNet18


# --------------------------
# Utility functions
# --------------------------
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


# --------------------------
# Main training / testing
# --------------------------
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

    train_dataset = datasets.ImageFolder(
        root=os.path.join(config['data_path'], 'train'),
        transform=transform_train
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join(config['data_path'], 'test'),
        transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4)
    )

    # --------------------------
    # Model
    # --------------------------
    model = ResNet18(num_classes=config['num_classes']).to(device)

    # --------------------------
    # Test only mode
    # --------------------------
    if config.get('test_only', False):
        model_path = os.path.join(config.get('save_path', '.'), 'best_baseline_model.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        acc = evaluate(model, test_loader, device)
        print(f"Test Accuracy: {acc:.4f}")
        return

    # --------------------------
    # Training setup
    # --------------------------
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['lr'],
        momentum=0.9,
        weight_decay=5e-4
    )

    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    epochs = config.get('epochs', 50)

    # --------------------------
    # Training loop
    # --------------------------
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        acc = evaluate(model, test_loader, device)

        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Loss: {avg_loss:.4f} "
              f"Test Accuracy: {acc:.4f}")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            os.makedirs(config.get('save_path', '.'), exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(config.get('save_path', '.'), 'best_baseline_model.pth'))
            print("Best baseline model updated!")

    print(f"\nTraining completed. Best Test Accuracy: {best_acc:.4f}")


# --------------------------
# Entry point
# --------------------------
if __name__ == '__main__':
    main()