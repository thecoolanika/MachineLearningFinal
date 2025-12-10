"""
Results comparison script.

This script compares the performance of all trained models (baseline CNN,
EfficientNetB0, ResNet50) and generates comprehensive comparison tables
and graphs.
"""

import sys
import argparse
from pathlib import Path
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def parse_results_file(file_path: Path):
    """
    Parse a results text file to extract metrics.
    
    Args:
        file_path: Path to results file
        
    Returns:
        Dictionary of metrics or None if file doesn't exist
    """
    if not file_path.exists():
        return None
    
    metrics = {}
    with open(file_path, 'r') as f:
        content = f.read()
        
        # Extract metrics using regex
        patterns = {
            'loss': r'Loss:\s*([\d.]+)',
            'accuracy': r'Accuracy:\s*([\d.]+)',
            'precision': r'Precision:\s*([\d.]+)',
            'recall': r'Recall:\s*([\d.]+)',
            'f1_score': r'F1-Score:\s*([\d.]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                metrics[key] = float(match.group(1))
    
    return metrics if metrics else None


def parse_yolov8_results(file_path: Path):
    """
    Parse YOLOv8 inference results.
    
    Args:
        file_path: Path to YOLOv8 results file
        
    Returns:
        Dictionary with inference metrics or None
    """
    if not file_path.exists():
        return None
    
    metrics = {}
    with open(file_path, 'r') as f:
        content = f.read()
        
        # Extract FPS
        fps_match = re.search(r'FPS:\s*([\d.]+)', content)
        if fps_match:
            metrics['fps'] = float(fps_match.group(1))
        
        # Extract average inference time
        avg_time_match = re.search(r'Average:\s*([\d.]+)\s*ms', content)
        if avg_time_match:
            metrics['avg_inference_time_ms'] = float(avg_time_match.group(1))
    
    return metrics if metrics else None


def collect_all_results(results_dir: str = 'results'):
    """
    Collect results from all model evaluation files.
    
    Args:
        results_dir: Directory containing results files
        
    Returns:
        Dictionary mapping model names to their metrics
    """
    results_path = Path(results_dir)
    all_results = {}
    
    # Classification models
    model_files = {
        'Baseline CNN': results_path / 'evaluation_results.txt',
        'EfficientNet-B0': results_path / 'efficientnet_b0_test_results.txt',
        'ResNet-50': results_path / 'resnet50_test_results.txt',
    }
    
    for model_name, file_path in model_files.items():
        metrics = parse_results_file(file_path)
        if metrics:
            all_results[model_name] = metrics
    
    # YOLOv8 (different format)
    yolov8_file = results_path / 'yolov8' / 'yolov8_inference_results.txt'
    yolov8_metrics = parse_yolov8_results(yolov8_file)
    if yolov8_metrics:
        all_results['YOLOv8'] = yolov8_metrics
    
    return all_results


def create_comparison_table(results: dict, output_path: Path):
    """
    Create a comparison table of all models.
    
    Args:
        results: Dictionary of model results
        output_path: Path to save the table
    """
    # Prepare data for DataFrame
    data = []
    for model_name, metrics in results.items():
        row = {'Model': model_name}
        
        # Classification metrics
        if 'accuracy' in metrics:
            row['Accuracy (%)'] = metrics['accuracy']
        if 'precision' in metrics:
            row['Precision'] = metrics['precision']
        if 'recall' in metrics:
            row['Recall'] = metrics['recall']
        if 'f1_score' in metrics:
            row['F1-Score'] = metrics['f1_score']
        if 'loss' in metrics:
            row['Loss'] = metrics['loss']
        
        # Inference metrics
        if 'fps' in metrics:
            row['FPS'] = metrics['fps']
        if 'avg_inference_time_ms' in metrics:
            row['Avg Inference Time (ms)'] = metrics['avg_inference_time_ms']
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    csv_path = output_path / 'model_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"Comparison table saved to {csv_path}")
    
    # Print table
    print("\n" + "=" * 80)
    print("Model Comparison Table")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    return df


def create_comparison_plots(results: dict, output_dir: Path):
    """
    Create comparison plots for model metrics.
    
    Args:
        results: Dictionary of model results
        output_dir: Directory to save plots
    """
    # Filter classification models
    classification_models = {k: v for k, v in results.items() 
                            if 'accuracy' in v and k != 'YOLOv8'}
    
    if not classification_models:
        print("No classification model results found for plotting.")
        return
    
    # Prepare data
    models = list(classification_models.keys())
    metrics_data = {
        'Accuracy (%)': [classification_models[m].get('accuracy', 0) for m in models],
        'Precision': [classification_models[m].get('precision', 0) for m in models],
        'Recall': [classification_models[m].get('recall', 0) for m in models],
        'F1-Score': [classification_models[m].get('f1_score', 0) for m in models],
    }
    
    # Create bar plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (metric_name, values) in enumerate(metrics_data.items()):
        ax = axes[idx]
        bars = ax.bar(models, values, color=['#3498db', '#2ecc71', '#e74c3c'][:len(models)])
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Comparison')
        ax.set_ylim(0, max(values) * 1.1 if max(values) > 0 else 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        # Rotate x-axis labels if needed
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    plt.tight_layout()
    plot_path = output_dir / 'model_comparison_metrics.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plots saved to {plot_path}")
    plt.close()
    
    # Create a combined radar/spider chart (simplified as bar chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(models))
    width = 0.2
    
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    normalized_values = {
        'Accuracy': [classification_models[m].get('accuracy', 0) / 100 for m in models],
        'Precision': [classification_models[m].get('precision', 0) for m in models],
        'Recall': [classification_models[m].get('recall', 0) for m in models],
        'F1-Score': [classification_models[m].get('f1_score', 0) for m in models],
    }
    
    for i, metric in enumerate(metric_names):
        offset = (i - 1.5) * width
        ax.bar([xi + offset for xi in x], normalized_values[metric], 
               width, label=metric)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score (Normalized)')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plot_path = output_dir / 'model_comparison_combined.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Combined comparison plot saved to {plot_path}")
    plt.close()


def create_summary_report(results: dict, output_path: Path):
    """
    Create a summary report of all results.
    
    Args:
        results: Dictionary of model results
        output_path: Path to save the report
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Wildfire Detection - Model Comparison Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("This report compares the performance of different models\n")
        f.write("trained for wildfire detection on the FlameVision dataset.\n\n")
        
        # Classification models
        classification_models = {k: v for k, v in results.items() 
                                if 'accuracy' in v and k != 'YOLOv8'}
        
        if classification_models:
            f.write("CLASSIFICATION MODELS\n")
            f.write("-" * 80 + "\n\n")
            
            for model_name, metrics in classification_models.items():
                f.write(f"{model_name}:\n")
                f.write(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.2f}%\n")
                f.write(f"  Precision: {metrics.get('precision', 'N/A'):.4f}\n")
                f.write(f"  Recall: {metrics.get('recall', 'N/A'):.4f}\n")
                f.write(f"  F1-Score: {metrics.get('f1_score', 'N/A'):.4f}\n")
                if 'loss' in metrics:
                    f.write(f"  Loss: {metrics.get('loss', 'N/A'):.4f}\n")
                f.write("\n")
            
            # Find best model for each metric
            f.write("BEST PERFORMING MODELS:\n")
            f.write("-" * 80 + "\n")
            
            if classification_models:
                best_acc = max(classification_models.items(), 
                              key=lambda x: x[1].get('accuracy', 0))
                best_f1 = max(classification_models.items(), 
                             key=lambda x: x[1].get('f1_score', 0))
                best_prec = max(classification_models.items(), 
                               key=lambda x: x[1].get('precision', 0))
                best_rec = max(classification_models.items(), 
                              key=lambda x: x[1].get('recall', 0))
                
                f.write(f"  Best Accuracy: {best_acc[0]} ({best_acc[1].get('accuracy', 0):.2f}%)\n")
                f.write(f"  Best F1-Score: {best_f1[0]} ({best_f1[1].get('f1_score', 0):.4f})\n")
                f.write(f"  Best Precision: {best_prec[0]} ({best_prec[1].get('precision', 0):.4f})\n")
                f.write(f"  Best Recall: {best_rec[0]} ({best_rec[1].get('recall', 0):.4f})\n")
                f.write("\n")
        
        # YOLOv8 results
        if 'YOLOv8' in results:
            f.write("OBJECT DETECTION MODEL (YOLOv8)\n")
            f.write("-" * 80 + "\n")
            yolov8_metrics = results['YOLOv8']
            f.write("YOLOv8 (Pre-trained, General Purpose):\n")
            if 'fps' in yolov8_metrics:
                f.write(f"  Average FPS: {yolov8_metrics['fps']:.2f}\n")
            if 'avg_inference_time_ms' in yolov8_metrics:
                f.write(f"  Average Inference Time: {yolov8_metrics['avg_inference_time_ms']:.2f} ms\n")
            f.write("\n")
            f.write("Note: YOLOv8 is a general-purpose object detection model.\n")
            f.write("For fire-specific detection, a custom-trained model would be needed.\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Report generated by compare_results.py\n")
        f.write("=" * 80 + "\n")
    
    print(f"Summary report saved to {output_path}")


def main():
    """Main function for results comparison."""
    parser = argparse.ArgumentParser(description='Compare model results')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing results files')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save comparison outputs')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("Collecting results from all models...")
    print("=" * 80)
    
    # Collect all results
    all_results = collect_all_results(str(results_dir))
    
    if not all_results:
        print("No results found! Please run evaluation scripts first.")
        print("Expected files:")
        print("  - results/evaluation_results.txt (Baseline CNN)")
        print("  - results/efficientnet_b0_test_results.txt (EfficientNet-B0)")
        print("  - results/resnet50_test_results.txt (ResNet-50)")
        print("  - results/yolov8/yolov8_inference_results.txt (YOLOv8)")
        return
    
    print(f"\nFound results for {len(all_results)} model(s):")
    for model_name in all_results.keys():
        print(f"  - {model_name}")
    
    # Create comparison table
    print("\nCreating comparison table...")
    df = create_comparison_table(all_results, output_dir)
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    create_comparison_plots(all_results, output_dir)
    
    # Create summary report
    print("\nCreating summary report...")
    create_summary_report(all_results, output_dir / 'model_comparison_report.txt')
    
    print("\n" + "=" * 80)
    print("Results comparison complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

