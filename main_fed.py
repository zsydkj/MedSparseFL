import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

from utils.options import load_config
from utils.sampling import non_iid_split
from models.Nets import ResNet18, GradientScoreNetwork
from options.federated_client import FederatedClient
from options.federated_server import FederatedServer
from options.support_utils import SupportConsistency

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
# Main federated training
# --------------------------
def main():
    # --------------------------
    # Load config
    # --------------------------
    config = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(config.get('seed', 42))

    # --------------------------
    # Prepare data
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4)
    )

    # --------------------------
    # Non-IID split for clients
    # --------------------------
    client_datasets = non_iid_split(train_dataset, num_clients=config['num_clients'], alpha=config['alpha'])

    # --------------------------
    # Initialize global model and GSN
    # --------------------------
    global_model = ResNet18(num_classes=config['num_classes']).to(device)
    gsn = GradientScoreNetwork(sum(p.numel() for p in global_model.parameters())).to(device)

    # --------------------------
    # Initialize clients
    # --------------------------
    clients = []
    support_trackers = []
    for i in range(config['num_clients']):
        client = FederatedClient(
            client_id=i,
            data=DataLoader(client_datasets[i], batch_size=config['batch_size'], shuffle=True),
            model=ResNet18(num_classes=config['num_classes']),
            lr=config['lr'],
            device=device,
            mask_ratio=config.get('mask_ratio', 0.1),
            gsn_threshold=config.get('gsn_threshold', 0.5)
        )
        clients.append(client)
        tracker = SupportConsistency(sum(p.numel() for p in client.model.parameters()))
        support_trackers.append(tracker)

    # --------------------------
    # Initialize federated server
    # --------------------------
    server = FederatedServer(clients, global_model, args=config)

    # --------------------------
    # Federated training rounds
    # --------------------------
    best_acc = 0.0
    for round_idx in range(config['rounds']):
        print(f"\n--- Round {round_idx + 1}/{config['rounds']} ---")

        # Broadcast global model to clients
        server.broadcast_model()

        # Each client performs local training
        for client in clients:
            client.local_train(gsn=gsn, epochs=config.get('local_epochs', 1))

        # Server aggregates updates
        server.aggregate()

        # Evaluate global model
        acc = evaluate(server.model, test_loader, device)
        print(f"Round {round_idx + 1} Test Accuracy: {acc:.4f}")

        # Save best global model
        if acc > best_acc:
            best_acc = acc
            save_path = config.get('save_path', '.')
            os.makedirs(save_path, exist_ok=True)
            torch.save(server.model.state_dict(), os.path.join(save_path, 'best_global_model.pth'))
            print("Best global model updated!")

    print(f"\nTraining completed. Best Test Accuracy: {best_acc:.4f}")

# --------------------------
# Entry point
# --------------------------
if __name__ == '__main__':
    main()