# Federated Learning with Sparse Gradient Optimization

This project implements a federated learning framework for image classification tasks, inspired by MedSparseFL.

It includes:
- Federated Learning (main_fed.py): Clients perform local training with sparse gradient updates. The server aggregates updates to improve a global model.
- Centralized Baseline (main_nn.py): Standard training of ResNet18 for comparison.
- Sparse Gradient Network (GSN) to identify important gradients for upload.
- Support for non-IID data distribution among clients.

Project Structure:
data/                # Dataset folder
models/              # Model definitions and federated classes
utils/               # Utility scripts (options, data sampling, etc.)
main_fed.py          # Federated learning training script
main_nn.py           # Centralized training script
_config.yml          # Configuration file
README.md
LICENSE

Requirements:
- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy

Usage:

Federated Learning:
python main_fed.py

Centralized Training:
python main_nn.py

Configuration:
Edit _config.yml to change hyperparameters like:
- num_clients
- batch_size
- rounds
- learning_rate
- data_path

License:
This project is licensed under the MIT License.  
You are free to use, modify, and distribute this code while keeping the copyright notice.