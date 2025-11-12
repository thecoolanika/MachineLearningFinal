# Wildfire Detection and Classification from Aerial Imagery

This project evaluates convolutional neural network models for wildfire detection and classification from aerial imagery. We train CNN models to perform binary classification (fire vs. non-fire) on the FlameVision dataset.

## Dataset

The FlameVision dataset contains 8,600 aerial images (5,000 fire images and 3,600 non-fire images) and is available at:
https://data.mendeley.com/datasets/fgvscdjsmt/4

The dataset includes train, validation, and test splits already provided.

## Models

We compare the performance of different CNN models:
- EfficientNetB0
- ResNet50
- YOLOv8
- Baseline CNN (simple architecture)

## Evaluation Metrics

Models are evaluated based on:
- Accuracy
- Precision
- Recall
- F1-score
- Inference time

## Project Structure

```
.
├── data/                    # Dataset directory (download here)
│   ├── train/
│   ├── val/
│   └── test/
├── models/                  # Saved model weights
├── scripts/                 # Training and evaluation scripts
├── src/                     # Source code
│   ├── data_loader.py      # Data loading utilities
│   ├── preprocessing.py    # Preprocessing and augmentation
│   ├── models/             # Model architectures
│   └── utils.py            # Utility functions
├── notebooks/              # Jupyter notebooks for EDA
├── results/                # Training results and plots
└── requirements.txt        # Python dependencies
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the Dataset
1. Visit https://data.mendeley.com/datasets/fgvscdjsmt/4
2. Download and extract the dataset to the `data/` directory
3. Verify the setup:
```bash
python scripts/download_dataset.py
```

### 3. Test Setup
```bash
python scripts/test_setup.py
```

### 4. Run Exploratory Data Analysis
```bash
python scripts/exploratory_analysis.py
```

### 5. Train Baseline Model
```bash
python scripts/train_baseline.py --epochs 50 --batch_size 32
```

### 6. Evaluate Model
```bash
python scripts/evaluate.py --model_path models/baseline_cnn_best.pth
```

For detailed setup instructions, see [SETUP.md](SETUP.md).

## Usage

### Training Baseline Model
```bash
python scripts/train_baseline.py --epochs 50 --batch_size 32 --lr 0.001
```

Options:
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--image_size`: Image size for training (default: 224)
- `--data_dir`: Path to dataset directory (default: data)

### Evaluation
```bash
python scripts/evaluate.py --model_path models/baseline_cnn_best.pth
```

### Exploratory Data Analysis
```bash
python scripts/exploratory_analysis.py
```

This generates:
- Dataset statistics
- Sample image visualizations
- Image size analysis

