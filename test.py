import os
import torch
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
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    return accuracy


# --------------------------
# Main testing function
# --------------------------
def main():
    config = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(config.get('seed', 42))

    # --------------------------
    # Prepare test dataset
    # --------------------------
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    test_dataset = datasets.ImageFolder(
        root=os.path.join(config['data_path'], 'test'),
        transform=transform_test
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4)
    )

    # --------------------------
    # Initialize model
    # --------------------------
    model = ResNet18(num_classes=config['num_classes']).to(device)

    # --------------------------
    # Choose model type to test
    # --------------------------
    model_type = config.get('model_type', 'baseline')
    # options: 'baseline' or 'federated'

    if model_type == 'baseline':
        model_path = os.path.join(
            config.get('save_path', '.'),
            'best_baseline_model.pth'
        )
    elif model_type == 'federated':
        model_path = os.path.join(
            config.get('save_path', '.'),
            'best_global_model.pth'
        )
    else:
        raise ValueError("model_type must be 'baseline' or 'federated'")

    # --------------------------
    # Load model weights
    # --------------------------
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from: {model_path}")

    # --------------------------
    # Evaluate
    # --------------------------
    acc = evaluate(model, test_loader, device)
    print(f"\nTest Accuracy ({model_type}): {acc:.4f}")


# --------------------------
# Entry point
# --------------------------
if __name__ == '__main__':
    main()