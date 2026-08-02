"""
==========================================================
Model Evaluation Module
==========================================================

PURPOSE:
    Evaluate the trained model on test/validation data using
    standard classification metrics.

METRICS EXPLAINED:
    - Accuracy:  What % of ALL predictions are correct?
    - Precision: Of items predicted as X, how many ARE X?
    - Recall:    Of actual X items, how many did we FIND?
    - F1-Score:  Harmonic mean of precision & recall
                 (balances both — most important metric!)
    
    WHY F1 OVER ACCURACY?
    If 90% of data is Neutral, a model predicting "Neutral" 
    for everything gets 90% accuracy but 0% on hate speech!
    F1-score catches this because recall for Hate Speech = 0.

CONFUSION MATRIX:
    Shows exactly WHERE the model makes mistakes:
    - Does it confuse Offensive with Hate Speech?
    - Does it miss Neutral content?
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import DEVICE, ID_TO_LABEL, MODEL_DIR, PROJECT_ROOT


def evaluate_model(model, data_loader, device=None):
    """
    Evaluate the model on a dataset and return all predictions.
    
    This function runs the model on all batches in the data_loader
    and collects predictions and true labels.
    
    Args:
        model: The trained BERT model
        data_loader: PyTorch DataLoader with test/validation data
        device: torch.device (GPU/CPU)
        
    Returns:
        tuple: (all_predictions, all_true_labels, all_probabilities)
    """
    if device is None:
        device = DEVICE
    
    model.eval()  # Set to evaluation mode
    
    all_predictions = []
    all_true_labels = []
    all_probabilities = []
    
    print("  Running evaluation...")
    
    with torch.no_grad():  # No gradients needed for evaluation
        for batch in tqdm(data_loader, desc="  Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get probabilities using softmax
            probabilities = torch.softmax(outputs.logits, dim=1)
            
            # Get predicted class (highest probability)
            predictions = torch.argmax(probabilities, dim=1)
            
            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return (
        np.array(all_predictions),
        np.array(all_true_labels),
        np.array(all_probabilities)
    )


def compute_metrics(predictions, true_labels):
    """
    Compute all classification metrics.
    
    Args:
        predictions (np.array): Predicted labels
        true_labels (np.array): Ground truth labels
        
    Returns:
        dict: Dictionary of all computed metrics
    """
    label_names = [ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL))]
    
    metrics = {
        # Overall accuracy
        'accuracy': accuracy_score(true_labels, predictions),
        
        # Precision: Of predicted Hate Speech, how many are correct?
        'precision_macro': precision_score(
            true_labels, predictions, average='macro', zero_division=0
        ),
        'precision_weighted': precision_score(
            true_labels, predictions, average='weighted', zero_division=0
        ),
        
        # Recall: Of actual Hate Speech, how many did we catch?
        'recall_macro': recall_score(
            true_labels, predictions, average='macro', zero_division=0
        ),
        'recall_weighted': recall_score(
            true_labels, predictions, average='weighted', zero_division=0
        ),
        
        # F1-Score: Balance of precision and recall
        'f1_macro': f1_score(
            true_labels, predictions, average='macro', zero_division=0
        ),
        'f1_weighted': f1_score(
            true_labels, predictions, average='weighted', zero_division=0
        ),
    }
    
    # Detailed classification report (per-class metrics)
    report = classification_report(
        true_labels, predictions,
        target_names=label_names,
        zero_division=0
    )
    
    metrics['classification_report'] = report
    
    return metrics


def print_metrics(metrics):
    """
    Display metrics in a formatted table.
    
    Args:
        metrics (dict): Dictionary from compute_metrics()
    """
    print("\n" + "=" * 60)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n  📊 Overall Metrics:")
    print(f"  {'─' * 40}")
    print(f"  Accuracy:            {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    print(f"  Precision (macro):   {metrics['precision_macro']:.4f}")
    print(f"  Precision (weighted):{metrics['precision_weighted']:.4f}")
    print(f"  Recall (macro):      {metrics['recall_macro']:.4f}")
    print(f"  Recall (weighted):   {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score (macro):    {metrics['f1_macro']:.4f}")
    print(f"  F1-Score (weighted): {metrics['f1_weighted']:.4f}")
    
    print(f"\n  📋 Detailed Classification Report:")
    print(f"  {'─' * 40}")
    print(metrics['classification_report'])


def plot_confusion_matrix(predictions, true_labels, save_path=None):
    """
    Create and save a beautiful confusion matrix heatmap.
    
    The confusion matrix shows:
    - Diagonal = correct predictions ✅
    - Off-diagonal = mistakes ❌
    
    Args:
        predictions (np.array): Predicted labels
        true_labels (np.array): True labels
        save_path (str): Path to save the plot (optional)
    """
    label_names = [ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL))]
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use a warm color scheme
    sns.heatmap(
        cm,
        annot=True,             # Show numbers in cells
        fmt='d',                # Integer format
        cmap='YlOrRd',          # Yellow-Orange-Red colormap
        xticklabels=label_names,
        yticklabels=label_names,
        square=True,
        linewidths=2,
        linecolor='white',
        cbar_kws={'label': 'Count'},
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix\nHate Speech Detection', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    if save_path is None:
        save_path = os.path.join(PROJECT_ROOT, "models", "confusion_matrix.png")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Confusion matrix saved to: {save_path}")
    
    plt.close()


def plot_training_history(history, save_path=None):
    """
    Plot training loss and validation accuracy over epochs.
    
    Args:
        history (dict): Training history from train.py
        save_path (str): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curve
    ax2.plot(epochs, history['val_accuracy'], 'g-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(PROJECT_ROOT, "models", "training_history.png")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Training history plot saved to: {save_path}")
    
    plt.close()


# ============================================
# RUN EVALUATION STANDALONE
# ============================================
if __name__ == "__main__":
    from transformers import BertForSequenceClassification, BertTokenizer
    from training.dataset import HateSpeechDataset
    from torch.utils.data import DataLoader
    
    print("Loading saved model for evaluation...")
    
    if os.path.exists(MODEL_DIR):
        tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
        model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
        model = model.to(DEVICE)
        
        print(f"Model loaded from: {MODEL_DIR}")
        print(f"Device: {DEVICE}")
    else:
        print(f"No saved model found at {MODEL_DIR}")
        print("Please run train.py first!")
