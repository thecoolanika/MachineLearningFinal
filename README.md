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
└── requirements.txt        # python3 dependencies
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
python3 scripts/download_dataset.py
```

### 3. Test Setup
```bash
python3 scripts/test_setup.py
```

### 4. Run Exploratory Data Analysis
```bash
python3 scripts/exploratory_analysis.py
```

### 5. Train Baseline Model
```bash
python3 scripts/train_baseline.py --epochs 5 --batch_size 64
```

### 6. Evaluate Model
```bash
python3 scripts/evaluate.py --model_path models/baseline_cnn_best.pth
```

For detailed setup instructions, see [SETUP.md](SETUP.md).

## Usage

### Training Models

#### Baseline CNN
```bash
python3 scripts/train_baseline.py --epochs 50 --batch_size 32 --lr 0.001
```

#### EfficientNet-B0
```bash
python3 scripts/train_efficientnet.py --epochs 5 --batch_size 64 --lr 1e-4
```

#### ResNet-50
```bash
python3 scripts/train_resnet50.py --epochs 50 --batch_size 64 --lr 1e-4
```

Common training options:
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 0.001 for baseline, 1e-4 for transfer learning)
- `--image_size`: Image size for training (default: 224)
- `--data_dir`: Path to dataset directory (default: data)
- `--use_augmentation`: Use data augmentation (default: True)
- `--no_pretrained`: Disable ImageNet pretraining (for transfer learning models)

### Evaluation

#### Baseline CNN
```bash
python3 scripts/evaluate.py --model_path models/baseline_cnn_best.pth
```

#### EfficientNet-B0
```bash
python3 scripts/evaluate_efficientnet.py --model_path models/efficientnet_b0_best.pth
```

#### ResNet-50
```bash
python3 scripts/evaluate_resnet50.py --model_path models/resnet50_best.pth
```

### YOLOv8 Object Detection

Run YOLOv8 inference for fire localization:
```bash
python3 scripts/yolov8_detection.py --split test --max_images 50
```

Options:
- `--split`: Dataset split to use (train/val/test, default: test)
- `--max_images`: Maximum number of images to process (default: 50)
- `--model_name`: YOLOv8 model name (default: yolov8n.pt)
- `--conf_threshold`: Confidence threshold for detections (default: 0.25)
- `--save_results`: Save visualization results (default: True)

### Results Comparison

Compare all trained models:
```bash
python3 scripts/compare_results.py
```

This generates:
- Comparison table (CSV)
- Comparison plots (metrics visualization)
- Summary report

### Exploratory Data Analysis
```bash
python3 scripts/exploratory_analysis.py
```

This generates:
- Dataset statistics
- Sample image visualizations
- Image size analysis

## Project Timeline

The project follows this timeline:

1. **Nov 6–9: Dataset setup and exploratory analysis**
   - Download and verify FlameVision dataset structure
   - Load sample images and understand data distribution
   - Implement data loading and preprocessing (resizing, normalization, augmentation)

2. **Nov 10–13: Baseline model training**
   - Implement and train simple CNN for binary classification
   - Record baseline accuracy and F1-score

3. **Nov 14–17: Transfer learning experiments**
   - Fine-tune EfficientNetB0 and ResNet50 (pretrained on ImageNet)
   - Compare performance to baseline
   - Save training curves and analyze overfitting

4. **Nov 18–21: Light object detection trial**
   - Use pre-trained YOLOv8 for basic fire localization
   - Run inference on subset of images
   - Evaluate qualitatively and measure inference time

5. **Nov 22–25: Results comparison and documentation**
   - Summarize results across all models
   - Create comparison graphs and tables
   - Document findings and conclusions

