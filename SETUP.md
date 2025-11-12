# Setup Instructions

This guide will help you set up the project and get started with wildfire detection.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- CUDA-capable GPU (optional, but recommended for training)

## Step 1: Install Dependencies

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Step 2: Download the Dataset

1. Visit the FlameVision dataset page:
   https://data.mendeley.com/datasets/fgvscdjsmt/4

2. Download the dataset (you may need to create a Mendeley account)

3. Extract the dataset to the `data` directory

4. The expected directory structure is:
```
data/
├── train/
│   ├── fire/
│   └── non_fire/
├── val/
│   ├── fire/
│   └── non_fire/
└── test/
    ├── fire/
    └── non_fire/
```

5. Verify the dataset setup:
```bash
python scripts/download_dataset.py
```

## Step 3: Run Exploratory Data Analysis

Before training, explore the dataset:

```bash
python scripts/exploratory_analysis.py
```

This will:
- Check dataset structure
- Display sample images
- Generate dataset statistics
- Analyze image sizes

Results will be saved in the `results/` directory.

## Step 4: Train Baseline Model

Train the baseline CNN model:

```bash
python scripts/train_baseline.py --epochs 50 --batch_size 32
```

Options:
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--image_size`: Image size for training (default: 224)
- `--data_dir`: Path to dataset directory (default: data)

The model will be saved in the `models/` directory, and training logs will be saved in the `logs/` directory.

## Step 5: Evaluate the Model

Evaluate the trained model on the test set:

```bash
python scripts/evaluate.py --model_path models/baseline_cnn_best.pth
```

This will generate:
- Test set metrics (accuracy, precision, recall, F1-score)
- Confusion matrix
- Evaluation results saved to `results/` directory

## Troubleshooting

### Dataset Not Found
If you get an error about the dataset not being found, make sure:
1. The dataset is downloaded and extracted
2. The directory structure matches the expected structure
3. Run `python scripts/download_dataset.py` to verify

### CUDA Out of Memory
If you encounter CUDA out of memory errors:
- Reduce the batch size: `--batch_size 16`
- Reduce the image size: `--image_size 128`
- Use CPU instead (slower): Set `CUDA_VISIBLE_DEVICES=""`

### Import Errors
If you encounter import errors:
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Make sure you're in the project root directory
- Try: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`

## Next Steps

After completing the baseline model training:
1. Compare with other models (EfficientNetB0, ResNet50, YOLOv8)
2. Experiment with different hyperparameters
3. Analyze model performance and inference time
4. Generate detailed evaluation reports

## Project Structure

```
.
├── data/                    # Dataset directory (download here)
├── models/                  # Saved model weights
├── scripts/                 # Training and evaluation scripts
│   ├── download_dataset.py
│   ├── exploratory_analysis.py
│   ├── train_baseline.py
│   └── evaluate.py
├── src/                     # Source code
│   ├── data_loader.py      # Data loading utilities
│   ├── preprocessing.py    # Preprocessing and augmentation
│   ├── models/             # Model architectures
│   │   ├── baseline_cnn.py
│   │   └── __init__.py
│   └── utils.py            # Utility functions
├── results/                # Training results and plots
├── logs/                   # Training logs (TensorBoard)
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
└── SETUP.md               # This file
```

