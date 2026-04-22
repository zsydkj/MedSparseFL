from __future__ import annotations

import argparse
from pathlib import Path

import torch

from main_fed import build_datasets, evaluate_multiclass, evaluate_multilabel, set_seed
from models.Nets import ResNet18
from utils.options import load_config



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model_type", type=str, choices=["baseline", "federated"], default=None)
    parser.add_argument("--model_path", type=str, default=None)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    config = load_config()
    if args.config is not None:
        from pathlib import Path as _Path
        import yaml

        with open(_Path(args.config), "r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
        if not isinstance(loaded, dict):
            raise ValueError("Configuration file must contain a YAML mapping")
        config = loaded

    set_seed(int(config.get("seed", 42)))
    device = "cuda" if torch.cuda.is_available() and config.get("use_cuda", True) else "cpu"
    _, test_dataset, task_type, num_classes = build_datasets(config)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
    )

    model = ResNet18(num_classes=num_classes).to(device)
    model_type = args.model_type or config.get("test_model_type", "federated")

    if args.model_path is not None:
        checkpoint_path = Path(args.model_path)
    else:
        save_dir = Path(config.get("save_dir", "./checkpoints"))
        checkpoint_name = (
            config.get("baseline_checkpoint", "best_baseline_model.pth")
            if model_type == "baseline"
            else config.get("federated_checkpoint", "best_global_model.pth")
        )
        checkpoint_path = save_dir / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    metrics = evaluate_multiclass(model, test_loader, device) if task_type == "multiclass" else evaluate_multilabel(model, test_loader, device)
    metric_name, metric_value = next(iter(metrics.items()))
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Test result ({model_type}) | {metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
