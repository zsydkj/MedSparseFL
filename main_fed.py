from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models.Nets import GradientScoreNetwork, ResNet18
from options.aggregation_utils import SecureAggregator
from options.federated_client import FederatedClient
from options.federated_server import FederatedServer
from options.privacy_utils import CountSketch, PairwiseMasking, PaillierAHE
from options.support_utils import HistoricalSupportTracker
from utils.options import load_config
from utils.sampling import non_iid_split


class ResizeToTensor:
    def __init__(self, image_size: tuple[int, int], random_horizontal_flip: bool = False) -> None:
        self.image_size = image_size
        self.random_horizontal_flip = random_horizontal_flip

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize(self.image_size)
        if self.random_horizontal_flip and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)


class DirectoryImageDataset(Dataset):
    def __init__(self, root: str, transform=None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.classes = sorted([path.name for path in self.root.iterdir() if path.is_dir()])
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.samples: list[tuple[str, int]] = []
        self.targets: list[int] = []
        for class_name in self.classes:
            for file_path in sorted((self.root / class_name).iterdir()):
                if file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    label = self.class_to_idx[class_name]
                    self.samples.append((str(file_path), label))
                    self.targets.append(label)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


class CSVImageDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_path_column: str, label_columns: Sequence[str], image_root: str | None, task_type: str, transform=None, image_id_suffix: str | None = None) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.image_path_column = image_path_column
        self.label_columns = list(label_columns)
        self.image_root = image_root
        self.task_type = task_type
        self.transform = transform
        self.image_id_suffix = image_id_suffix
        labels = self.dataframe[self.label_columns].values
        self.targets = labels.astype(np.float32 if task_type == "multilabel" else np.int64)

    def __len__(self) -> int:
        return len(self.dataframe)

    def _resolve_path(self, row: pd.Series) -> str:
        raw_value = str(row[self.image_path_column])
        if self.image_root is None:
            return raw_value
        candidate = Path(self.image_root) / raw_value
        if candidate.exists():
            return str(candidate)
        if self.image_id_suffix is not None:
            candidate = Path(self.image_root) / f"{raw_value}{self.image_id_suffix}"
            if candidate.exists():
                return str(candidate)
        return str(Path(self.image_root) / raw_value)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image = Image.open(self._resolve_path(row))
        if self.transform is not None:
            image = self.transform(image)
        if self.task_type == "multiclass":
            label = torch.tensor(int(row[self.label_columns[0]]), dtype=torch.long)
        else:
            label = torch.tensor(row[self.label_columns].values.astype(np.float32), dtype=torch.float32)
        return image, label


class Ham10000Dataset(CSVImageDataset):
    def __init__(self, dataframe: pd.DataFrame, image_root: str, diagnosis_column: str = "dx", image_id_column: str = "image_id", transform=None) -> None:
        classes = sorted(dataframe[diagnosis_column].astype(str).unique().tolist())
        class_to_index = {name: index for index, name in enumerate(classes)}
        prepared = dataframe.copy()
        prepared["label"] = prepared[diagnosis_column].astype(str).map(class_to_index).astype(int)
        super().__init__(prepared, image_id_column, ["label"], image_root, "multiclass", transform=transform, image_id_suffix=".jpg")
        self.classes = classes


class CheXpertDataset(CSVImageDataset):
    def __init__(self, dataframe: pd.DataFrame, image_root: str | None, label_columns: Sequence[str], path_column: str = "Path", uncertain_value: float = 0.0, transform=None) -> None:
        prepared = dataframe.copy()
        for column in label_columns:
            prepared[column] = prepared[column].fillna(0.0).replace(-1.0, uncertain_value)
        super().__init__(prepared, path_column, label_columns, image_root, "multilabel", transform=transform)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_multiclass(model: torch.nn.Module, dataloader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        predictions = model(inputs).argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)
    return {"accuracy": correct / max(total, 1)}


@torch.no_grad()
def evaluate_multilabel(model: torch.nn.Module, dataloader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    tp = None
    fp = None
    fn = None
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device).float()
        predictions = (torch.sigmoid(model(inputs)) >= 0.5).float()
        batch_tp = (predictions * targets).sum(dim=0)
        batch_fp = (predictions * (1.0 - targets)).sum(dim=0)
        batch_fn = ((1.0 - predictions) * targets).sum(dim=0)
        if tp is None:
            tp, fp, fn = batch_tp, batch_fp, batch_fn
        else:
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"macro_f1": float(f1.mean().item())}


def build_datasets(config: dict):
    dataset_type = config["dataset_type"].lower()
    image_size = tuple(config.get("image_size", [224, 224]))
    train_transform = ResizeToTensor(image_size, random_horizontal_flip=True)
    eval_transform = ResizeToTensor(image_size, random_horizontal_flip=False)

    if dataset_type == "imagefolder":
        train_dataset = DirectoryImageDataset(config["train_dir"], transform=train_transform)
        test_dataset = DirectoryImageDataset(config["test_dir"], transform=eval_transform)
        return train_dataset, test_dataset, "multiclass", len(train_dataset.classes)
    if dataset_type == "ham10000":
        metadata = pd.read_csv(config["metadata_csv"])
        split_column = config.get("split_column")
        if split_column is None:
            train_df = pd.read_csv(config["train_csv"])
            test_df = pd.read_csv(config["test_csv"])
        else:
            train_df = metadata[metadata[split_column] == config.get("train_split_name", "train")].reset_index(drop=True)
            test_df = metadata[metadata[split_column] == config.get("test_split_name", "test")].reset_index(drop=True)
        train_dataset = Ham10000Dataset(train_df, config["image_root"], diagnosis_column=config.get("diagnosis_column", "dx"), image_id_column=config.get("image_id_column", "image_id"), transform=train_transform)
        test_dataset = Ham10000Dataset(test_df, config["image_root"], diagnosis_column=config.get("diagnosis_column", "dx"), image_id_column=config.get("image_id_column", "image_id"), transform=eval_transform)
        return train_dataset, test_dataset, "multiclass", len(train_dataset.classes)
    if dataset_type == "chexpert":
        label_columns = config["label_columns"]
        train_df = pd.read_csv(config["train_csv"])
        valid_path = config.get("valid_csv", config.get("test_csv"))
        if valid_path is None:
            raise ValueError("chexpert configuration requires valid_csv or test_csv")
        test_df = pd.read_csv(valid_path)
        train_dataset = CheXpertDataset(train_df, config.get("image_root"), label_columns, path_column=config.get("path_column", "Path"), uncertain_value=float(config.get("uncertain_value", 0.0)), transform=train_transform)
        test_dataset = CheXpertDataset(test_df, config.get("image_root"), label_columns, path_column=config.get("path_column", "Path"), uncertain_value=float(config.get("uncertain_value", 0.0)), transform=eval_transform)
        return train_dataset, test_dataset, "multilabel", len(label_columns)
    if dataset_type in {"csv_multiclass", "csv_multilabel"}:
        task_type = "multiclass" if dataset_type == "csv_multiclass" else "multilabel"
        train_df = pd.read_csv(config["train_csv"])
        test_df = pd.read_csv(config["test_csv"])
        train_dataset = CSVImageDataset(train_df, config["image_path_column"], config["label_columns"], config.get("image_root"), task_type, transform=train_transform)
        test_dataset = CSVImageDataset(test_df, config["image_path_column"], config["label_columns"], config.get("image_root"), task_type, transform=eval_transform)
        num_classes = len(config["label_columns"]) if task_type == "multilabel" else int(max(train_df[config["label_columns"][0]].max(), test_df[config["label_columns"][0]].max()) + 1)
        return train_dataset, test_dataset, task_type, num_classes
    raise ValueError(f"Unsupported dataset_type: {config['dataset_type']}")


def main() -> None:
    config = load_config()
    set_seed(int(config.get("seed", 42)))
    device = "cuda" if torch.cuda.is_available() and config.get("use_cuda", True) else "cpu"
    train_dataset, test_dataset, task_type, num_classes = build_datasets(config)
    client_subsets = non_iid_split(train_dataset, int(config["num_clients"]), float(config["dirichlet_alpha"]), seed=int(config.get("seed", 42)))
    test_loader = DataLoader(test_dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config.get("num_workers", 0)))
    global_model = ResNet18(num_classes=num_classes).to(device)
    gsn = GradientScoreNetwork(client_feature_dim=1 + int(config["num_clients"])).to(device)
    gsn_optimizer = torch.optim.Adam(gsn.parameters(), lr=float(config["gsn_lr"]))
    count_sketch = CountSketch(int(config["count_sketch_num_hash"]), int(config["count_sketch_size"]), int(config["count_sketch_seed"]))
    masking = PairwiseMasking(int(config["mask_master_seed"]), int(config["mask_bound"]))
    he = PaillierAHE(key_bits=int(config["paillier_key_bits"]))
    support_tracker = HistoricalSupportTracker(sum(parameter.numel() for parameter in global_model.parameters()), float(config["target_upload_ratio"]), int(config["support_window_size"]), float(config["support_epsilon"]), float(config["support_min_frequency"]))
    aggregator = SecureAggregator(global_model, count_sketch, he, support_tracker, int(config["quantization_scale_exp"]), device)
    clients = []
    for client_id, subset in enumerate(client_subsets):
        loader = DataLoader(subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=int(config.get("num_workers", 0)))
        clients.append(FederatedClient(client_id, loader, ResNet18(num_classes=num_classes), gsn, gsn_optimizer, float(config["lr"]), device, int(config["num_clients"]), len(subset), task_type, float(config["target_upload_ratio"]), float(config["ema_decay"]), float(config["gsn_reg_lambda"]), count_sketch, masking, he, int(config["quantization_scale_exp"]), int(config["gsn_block_size"])))
    server = FederatedServer(clients, global_model, aggregator, float(config["client_fraction"]), device)
    save_dir = Path(config.get("save_dir", "./checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    best_metric = -1.0
    for round_idx in range(int(config["rounds"])):
        stats = server.train_round(round_idx, int(config["local_epochs"]))
        metrics = evaluate_multiclass(global_model, test_loader, device) if task_type == "multiclass" else evaluate_multilabel(global_model, test_loader, device)
        metric_name, metric_value = next(iter(metrics.items()))
        print(f"Round {round_idx + 1:03d} | {metric_name}: {metric_value:.4f} | clients: {int(stats['num_selected_clients'])} | density: {stats['mean_upload_density']:.4f} | consistency: {stats['support_consistency']:.4f}")
        if metric_value > best_metric:
            best_metric = metric_value
            torch.save(global_model.state_dict(), save_dir / "best_global_model.pth")
            torch.save(gsn.state_dict(), save_dir / "best_gsn.pth")
    print(f"Best metric: {best_metric:.4f}")


if __name__ == "__main__":
    main()
