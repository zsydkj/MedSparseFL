# MedSparseFL

Privacy-Preserving Support-Aware Sparse Federated Learning for Medical
Imaging

------------------------------------------------------------------------

# 1 Project Overview

## Introduction

MedSparseFL is a privacy-preserving federated learning framework
designed for heterogeneous (non-IID) medical data.\
The method introduces a Support-aware Gradient Sparsification mechanism and a
Stability-aware Robust Aggregation strategy to mitigate gradient drift
and aggregation bias under medical data heterogeneity.

The framework significantly reduces communication overhead while
maintaining convergence stability and privacy guarantees.

------------------------------------------------------------------------

## Key Features

-   Support-aware Gradient Sparsification (GSN-based)
-   Stability-aware Robust Aggregation
-   Dual-layer Privacy Protection (AHE + Secure Aggregation)
-   Sparse Communication with Residual Feedback
-   Modular and extensible FL pipeline

------------------------------------------------------------------------

## Target Scenarios

-   Multi-institutional medical image classification
-   Privacy-sensitive healthcare AI training
-   Communication-constrained federated systems
-   Heterogeneous non-IID federated environments

------------------------------------------------------------------------

# 2 Quick Start

## Environment Requirements

-   Python ≥ 3.9
-   PyTorch ≥ 2.0
-   CUDA (optional)
-   NumPy
-   PyYAML
-   tqdm
-   matplotlib

------------------------------------------------------------------------

## Installation

``` bash
git clone https://github.com/your-repo/MedSparseFL.git
cd MedSparseFL
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Minimal Running Example

Run federated training:

``` bash
python main_fed.py --config config.yml
```

Train centralized baseline:

``` bash
python main_nn.py
```

Test model:

``` bash
python test.py --model_path saved_model.pth
```

------------------------------------------------------------------------

# 3 Project Structure

    MedSparseFL/
    │
    ├── data/
    ├── models/
    │   ├── _init_.py
    │   ├── Nets.py
    │   ├── Fed.py
    │   ├── Update.py
    ├── options/
    │   ├── _init_.py
    │   ├── federated_client.py
    │   ├── federated_server.py
    │   ├── aggregation_utils.py
    │   ├── privacy_utils.py
    │   ├── support_utils.py
    ├── utils/
    │   ├── sampling.py
    │   ├── options.py
    ├── main_fed.py
    ├── main_nn.py
    ├── test.py
    ├── config.yml
    └── README.md

------------------------------------------------------------------------

# 4 Configuration Guide

## config.yml Example

``` yaml
training:
  rounds: 100
  local_epochs: 5
  batch_size: 32
  lr: 0.01
  momentum: 0.9

federation:
  num_clients: 10
  fraction: 0.5
  noniid: true

sparsification:
  sparsity_ratio: 0.1
  residual: true

privacy:
  homomorphic: true
  scheme: paillier
  key_size: 2048

aggregation:
  robust: true
  stability_threshold: 0.2
```

------------------------------------------------------------------------

## Important Parameters

  Parameter             Description
  --------------------- ----------------------------------
  rounds                Total global rounds
  local_epochs          Local client epochs
  sparsity_ratio        Percentage of selected gradients
  clip_norm             Gradient clipping bound
  stability_threshold   Anomaly suppression threshold

------------------------------------------------------------------------

# 5 Usage

## Train Federated Model

``` bash
python main_fed.py --config config.yml
```

### Training Pipeline

1.  Server initializes global model
2.  Clients perform local SGD
3.  Gradient clipping
4.  Gradient Score Network scoring
5.  Sparse support selection
6.  Residual feedback update
7.  Privacy enhancement
8.  Secure masked aggregation
9.  Stability-aware robust aggregation
10. Global model update

------------------------------------------------------------------------

## Train Centralized Model

``` bash
python main_nn.py
```

------------------------------------------------------------------------

## Test Model

``` bash
python test.py --model_path model.pth
```

Outputs:

-   Test accuracy
-   Test loss

------------------------------------------------------------------------

# 6 Core Modules Explanation

## Data Module

-   Supports IID and non-IID partition
-   Dirichlet-based heterogeneous splitting
-   Multi-label medical compatibility

------------------------------------------------------------------------

## Model Module

-   ResNet backbone
-   Gradient Score Network (GSN)
-   Stable support estimation across rounds

------------------------------------------------------------------------

## Client Logic

Client-side operations:

-   Local training
-   Gradient clipping
-   Sparse support selection
-   Residual accumulation
-   Privacy protection mechanism

------------------------------------------------------------------------

## Server Logic

Server-side operations:

-   Support frequency tracking
-   Stability-based anomaly suppression
-   Robust weighted aggregation

------------------------------------------------------------------------

## Privacy Protection

Dual-layer privacy:

1.  Additive homomorphic encryption
2.  Secure Aggregation masking

------------------------------------------------------------------------

# 7 Experimental Setup

## Datasets

-   CheXpert
-   HAM10000

------------------------------------------------------------------------

# 8 License

MIT License

Copyright (c) 2026

This project is licensed under the MIT license. For details, please refer to the LICENSE file.