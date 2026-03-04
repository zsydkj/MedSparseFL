import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from utils.options import load_config
from utils.sampling import non_iid_split
from models.Nets import ResNet18, GradientScoreNetwork
from models.Fed import Server, Client
import copy
import os


# ------------------------------
# Helper functions
# ------------------------------
def evaluate_model(model, dataloader, device='cuda'):
    """Evaluate the accuracy of the model on the given dataloader"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            preds = output.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------
# Main federated learning process
# ------------------------------
def main():
    # Load configuration file
    config = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(config.get('seed', 42))

    # --------------------------
    # Load dataset
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

    # Split dataset into non-IID partitions for clients
    client_datasets = non_iid_split(train_dataset, num_clients=config['num_clients'], alpha=config['alpha'])

    # --------------------------
    # Initialize global model and GSN
    # --------------------------
    global_model = ResNet18(num_classes=config['num_classes']).to(device)
    gsn = GradientScoreNetwork(input_dim=global_model.model.fc.in_features).to(device)

    # Initialize clients
    clients = []
    for i, ds in enumerate(client_datasets):
        dataloader = DataLoader(ds, batch_size=config['batch_size'], shuffle=True)
        client = Client(client_id=i, model=global_model, dataloader=dataloader, device=device)
        clients.append(client)

    # Initialize server
    server = Server(global_model)

    # --------------------------
    # Training loop
    # --------------------------
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    best_acc = 0.0
    for r in range(config['rounds']):
        print(f"\n=== Round {r + 1}/{config['rounds']} ===")

        # Each client computes sparse gradient update
        client_updates = []
        for client in clients:
            update = client.compute_sparse_update(gsn)
            client_updates.append(update)

        # Server aggregates client updates
        server.aggregate_updates(client_updates)

        # Evaluate global model
        acc = evaluate_model(server.global_model, test_loader, device)
        print(f"Test Accuracy after round {r + 1}: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(server.global_model.state_dict(), 'best_global_model.pth')
            print("Best model updated!")

    print(f"\nTraining completed. Best Test Accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    main()
    main()