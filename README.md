# MedSparseFL

Privacy-Preserving Support-Aware Sparse Federated Learning for Medical Imaging

------------------------------------------------------------------------

# 1 Project Overview

## Introduction

MedSparseFL is a privacy-preserving federated learning framework designed for heterogeneous (non-IID) medical data.
The implementation integrates support-aware sparse local updates, sketch-domain secure aggregation, additive homomorphic encryption, and server-side support consistency reweighting.

To keep the method computationally tractable, the current implementation adopts block-level Gradient Score Network (GSN) gating rather than full coordinate-level scoring.
Across the federated pipeline, the uploaded object is defined consistently as a weighted sparse local model delta.

------------------------------------------------------------------------

## Key Features

- Support-aware sparse local update generation with block-level GSN
- Residual compensation for sparse local model deltas
- Count Sketch compression in the sketch domain
- Pairwise masking and Paillier additive homomorphic aggregation
- Server-side support consistency stabilization
- Modular federated learning pipeline with compatibility-layer `models/Fed.py`

------------------------------------------------------------------------

## Target Scenarios

- Multi-institutional medical image classification
- Privacy-sensitive healthcare AI training
- Communication-constrained federated systems
- Heterogeneous non-IID federated environments

------------------------------------------------------------------------

# 2 Quick Start

## Environment Requirements

- Python 3.9 or newer
- PyTorch 2.0 or newer
- NumPy
- pandas
- Pillow
- PyYAML

------------------------------------------------------------------------

## Installation

```bash
git clone https://github.com/your-repo/MedSparseFL.git
cd MedSparseFL
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Minimal Running Example

Run federated training:

```bash
python main_fed.py --config config.yml
```

Train the centralized baseline:

```bash
python main_nn.py --config config.yml
```

Evaluate the saved federated checkpoint defined in the config:

```bash
python test.py --config config.yml --model_type federated
```

Evaluate the saved baseline checkpoint defined in the config:

```bash
python test.py --config config.yml --model_type baseline
```

Evaluate an explicit checkpoint path:

```bash
python test.py --config config.yml --model_path ./checkpoints/best_global_model.pth
```

------------------------------------------------------------------------

# 3 Project Structure

```text
MedSparseFL/
├── data/
│   ├── __init__.py
│   └── README.md
├── models/
│   ├── __init__.py
│   ├── Fed.py
│   ├── Nets.py
│   └── Update.py
├── options/
│   ├── __init__.py
│   ├── aggregation_utils.py
│   ├── federated_client.py
│   ├── federated_server.py
│   ├── privacy_utils.py
│   └── support_utils.py
├── utils/
│   ├── __init__.py
│   ├── options.py
│   └── sampling.py
├── config.yml
├── main_fed.py
├── main_nn.py
└── test.py
```

------------------------------------------------------------------------

# 4 Configuration Guide

## config.yml Overview

The main configuration file is `config.yml`.
It controls dataset loading, task type selection, federated optimization, centralized baseline training, privacy modules, support consistency, checkpoint naming, and test behavior.

------------------------------------------------------------------------

## Core Configuration Fields

### Dataset and task settings

- `dataset_type`: dataset loader type. Supported values are `imagefolder`, `ham10000`, `chexpert`, `csv_multiclass`, and `csv_multilabel`.
- `task_type`: task formulation. Supported values are `multiclass` and `multilabel`.
- `num_classes`: number of prediction targets.
- `data_root`: base dataset directory.
- `train_dir`, `test_dir`: image-folder style train and test directories when applicable.
- `train_csv`, `test_csv`: CSV annotation files for CSV-based datasets.
- `image_root`: image root directory used by CSV-based datasets.

### Federated training settings

- `rounds`: total communication rounds.
- `num_clients`: total number of clients.
- `clients_per_round`: number of selected clients per round.
- `local_epochs`: local training epochs per selected client.
- `batch_size`: mini-batch size.
- `lr`: local learning rate.

### Sparse update and GSN settings

- `block_size`: number of parameters per block for GSN scoring.
- `target_sparsity`: target active upload ratio after probability recalibration.
- `residual`: whether residual compensation is enabled.
- `gsn_hidden_dim`: hidden size of the block-level GSN.
- `gsn_lr`: optimizer learning rate for GSN updates.
- `gsn_lambda_budget`: sparsity budget regularization coefficient.

### Count Sketch and privacy settings

- `sketch_num_hash`: number of Count Sketch hash rows.
- `sketch_size`: bucket size per hash row.
- `quant_scale`: fixed-point quantization scale before encryption.
- `paillier_key_bits`: Paillier key length used by the current implementation.
- `mask_seed`: global seed used to derive pairwise masking seeds.

### Support consistency settings

- `support_window_size`: number of rounds retained in the historical support window.
- `support_topk`: number of recovered coordinates used to define support.
- `support_mix`: interpolation weight for support-based stabilization.

### Checkpoint and evaluation settings

- `save_dir`: output directory for checkpoints.
- `baseline_ckpt_name`: checkpoint filename for the centralized baseline.
- `federated_ckpt_name`: checkpoint filename for the federated model.
- `model_type`: default test target in `test.py`, either `baseline` or `federated`.

------------------------------------------------------------------------

# 5 Usage

## Train Federated Model

```bash
python main_fed.py --config config.yml
```

Federated training follows the unified update semantics below:

1. The server broadcasts the current global model.
2. Each selected client performs local training from the broadcast model.
3. The client computes its local model delta with respect to the broadcast global model.
4. Residual compensation is applied to the local delta.
5. Block-level GSN estimates upload probabilities for parameter blocks.
6. The sparse local model delta is compressed with Count Sketch.
7. Pairwise masking and Paillier encryption are applied in the sketch domain.
8. The server aggregates encrypted sketches, decrypts the result, recovers the update, applies support-consistency stabilization, and updates the global model.

Saved checkpoint:
- `save_dir/federated_ckpt_name`

------------------------------------------------------------------------

## Train Centralized Model

```bash
python main_nn.py --config config.yml
```

The centralized baseline uses the same dataset builder and task-type logic as the federated script, so multiclass and multilabel settings remain consistent across both entry points.

Saved checkpoint:
- `save_dir/baseline_ckpt_name`

------------------------------------------------------------------------

## Test Model

Use the model type defined in the config:

```bash
python test.py --config config.yml --model_type federated
python test.py --config config.yml --model_type baseline
```

Or evaluate an explicit checkpoint:

```bash
python test.py --config config.yml --model_path ./checkpoints/best_global_model.pth
```

Outputs depend on `task_type`:
- multiclass tasks report accuracy
- multilabel tasks report macro-F1 with sigmoid threshold 0.5

------------------------------------------------------------------------

# 6 Core Modules Explanation

## Data Module

`utils/sampling.py` handles client partitioning for federated training.
The codebase supports `imagefolder`, `ham10000`, `chexpert`, `csv_multiclass`, and `csv_multilabel` through the shared dataset-building path used by `main_fed.py`, `main_nn.py`, and `test.py`.

------------------------------------------------------------------------

## Model Module

`models/Nets.py` contains the backbone classifier and the block-level Gradient Score Network.
The GSN receives block statistics and client-conditioned features, then outputs upload probabilities for parameter blocks.

------------------------------------------------------------------------

## Update Module

`models/Update.py` provides utilities for flattening model deltas, building block-level GSN features, applying probability recalibration, and generating sparse updates with residual compensation and straight-through estimation.

------------------------------------------------------------------------

## Client Logic

`options/federated_client.py` performs local training, constructs weighted local model deltas, applies GSN-based sparsification, maintains residual states, and transforms sparse updates into encrypted sketch-domain uploads.

------------------------------------------------------------------------

## Server Logic

`options/federated_server.py` and `options/aggregation_utils.py` handle ciphertext aggregation, sketch recovery, normalized weighted updating, and support-consistency stabilization.
The server update path is aligned with sparse local model deltas rather than raw gradients.

------------------------------------------------------------------------

## Privacy Protection

`options/privacy_utils.py` implements Count Sketch compression, pairwise masking, fixed-point quantization, and Paillier additive homomorphic aggregation.
The current Paillier implementation is suitable for method alignment and small-scale experiments rather than large-scale encrypted deployment.

------------------------------------------------------------------------

# 7 Experimental Setup

## Supported Datasets

- CheXpert
- HAM10000
- Generic image-folder multiclass datasets
- CSV-based multiclass datasets
- CSV-based multilabel datasets

See `data/README.md` for the required directory layouts and CSV field expectations.

------------------------------------------------------------------------

# 8 License

MIT License

Copyright (c) 2026

This project is licensed under the MIT License. See the LICENSE file for details.
