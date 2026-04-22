from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from main_fed import build_datasets, evaluate_multiclass, evaluate_multilabel, set_seed
from models.Nets import ResNet18
from utils.options import load_config



def build_criterion(task_type: str) -> nn.Module:
    if task_type == "multilabel":
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()



def main() -> None:
    config = load_config()
    set_seed(int(config.get("seed", 42)))
    device = "cuda" if torch.cuda.is_available() and config.get("use_cuda", True) else "cpu"

    train_dataset, test_dataset, task_type, num_classes = build_datasets(config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
    )

    model = ResNet18(num_classes=num_classes).to(device)
    criterion = build_criterion(task_type)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config["lr"]),
        momentum=float(config.get("momentum", 0.9)),
        weight_decay=float(config.get("weight_decay", 5e-4)),
    )

    save_dir = Path(config.get("save_dir", "./checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = config.get("baseline_checkpoint", "best_baseline_model.pth")
    epochs = int(config.get("baseline_epochs", config.get("epochs", 50)))

    if bool(config.get("baseline_test_only", False)):
        checkpoint_path = save_dir / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Baseline checkpoint not found: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        metrics = evaluate_multiclass(model, test_loader, device) if task_type == "multiclass" else evaluate_multilabel(model, test_loader, device)
        metric_name, metric_value = next(iter(metrics.items()))
        print(f"Baseline test | {metric_name}: {metric_value:.4f}")
        return

    best_metric = -1.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if task_type == "multiclass":
                targets = targets.long()
            else:
                targets = targets.float()

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        metrics = evaluate_multiclass(model, test_loader, device) if task_type == "multiclass" else evaluate_multilabel(model, test_loader, device)
        metric_name, metric_value = next(iter(metrics.items()))
        epoch_loss = running_loss / max(total_samples, 1)
        print(f"Epoch {epoch + 1:03d} | loss: {epoch_loss:.4f} | {metric_name}: {metric_value:.4f}")

        if metric_value > best_metric:
            best_metric = metric_value
            torch.save(model.state_dict(), save_dir / checkpoint_name)

    print(f"Best baseline metric: {best_metric:.4f}")


if __name__ == "__main__":
    main()
