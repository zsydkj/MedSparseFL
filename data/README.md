Data Folder Readme
==================

This folder is intended to store the datasets used in the MedSparseFL project. 
Due to size and licensing restrictions, the datasets are not included in this repository. 
Users should download them from the official sources.

Supported Datasets
------------------

1. CheXpert
   - Description: Chest X-ray dataset for multi-label classification (pneumonia, effusion, etc.)
   - Official link: https://stanfordmlgroup.github.io/competitions/chexpert/
   - Structure after download:
     CheXpert/
     ├── images/
     │   ├── patient_001.png
     │   ├── patient_002.png
     │   └── ...
     └── labels.csv
         # CSV contains image filenames and corresponding multi-label annotations

2. HAM10000
   - Description: Skin lesion dataset for multi-class classification (7 skin lesion types)
   - Official link: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
   - Structure after download:
     HAM10000/
     ├── images/
     │   ├── ISIC_001.png
     │   ├── ISIC_002.png
     │   └── ...
     └── labels.csv
         # CSV contains image filenames and corresponding class labels

Instructions
------------

1. Download the datasets from their respective official sources.
2. Organize them under the `data/` folder following the structure above.
3. Ensure that the CSV label files match the image filenames exactly.
4. The training scripts (`main_fed.py` and `main_nn.py`) will automatically read data from this folder.

Notes
-----

- Make sure to comply with the dataset licenses and usage policies.
- Do not share the datasets directly in public repositories due to licensing restrictions.
- You can preprocess the datasets as needed for your experiments.