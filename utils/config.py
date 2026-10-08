"""
==========================================================
Configuration File for Hate Speech Detection Project
==========================================================

This file contains ALL configurable parameters for the 
entire project — model settings, file paths, hyperparameters,
and label mappings.

WHY A CENTRAL CONFIG?
- Avoids hardcoding values across multiple files
- Makes it easy to tweak settings without hunting through code
- Professional software engineering practice
"""

import os
import torch

# ============================================
# 1. PROJECT PATHS
# ============================================
# os.path.dirname gets the directory of the current file,
# then we go one level up to get the project root.

# Root directory of the project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SAMPLE_DATA_PATH = os.path.join(DATA_DIR, "sample_data.csv")

# Model save directory
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "saved_model")

# ============================================
# 2. MODEL CONFIGURATION
# ============================================
# We use Multilingual BERT (mBERT) which supports 104 languages
# including Hindi and English — perfect for Hinglish text!

MODEL_NAME = "bert-base-multilingual-cased"  # Hugging Face model identifier
NUM_LABELS = 3                                # Hate Speech, Offensive, Neutral
MAX_SEQ_LENGTH = 64                          # Max tokens per input text (64 is optimal & fast)

# ============================================
# 3. LABEL MAPPINGS
# ============================================
# These define how we convert between human-readable labels
# and numeric IDs that the model uses internally.

LABEL_TO_ID = {
    "Neutral": 0,
    "Offensive": 1,
    "Hate Speech": 2
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# Color codes for UI display
LABEL_COLORS = {
    "Neutral": "#2ecc71",       # Green
    "Offensive": "#f39c12",     # Orange/Yellow
    "Hate Speech": "#e74c3c"    # Red
}

LABEL_EMOJIS = {
    "Neutral": "🟢",
    "Offensive": "🟡", 
    "Hate Speech": "🔴"
}

# ============================================
# 4. TRAINING HYPERPARAMETERS
# ============================================
# These control HOW the model learns:
# - LEARNING_RATE: How big each learning step is (too big = unstable, too small = slow)
# - BATCH_SIZE: How many samples to process at once (limited by GPU memory)
# - NUM_EPOCHS: How many times to go through the entire dataset
# - WARMUP_STEPS: Gradually increase learning rate at the start to stabilize training

LEARNING_RATE = 3e-5        # Standard fine-tuning learning rate
BATCH_SIZE = 16             # Faster CPU batching
NUM_EPOCHS = 3              # 3 epochs is optimal for fine-tuning
WARMUP_STEPS = 50           # Warmup steps
WEIGHT_DECAY = 0.01         # L2 regularization to prevent overfitting
TRAIN_SPLIT = 0.85          # 85% train, 15% validation
EARLY_STOPPING_PATIENCE = 3 # Stop if no improvement for 3 epochs

# ============================================
# 5. DEVICE CONFIGURATION
# ============================================
# Automatically detect if a GPU is available.
# GPU training is ~10x faster than CPU for transformer models.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# 6. LIME EXPLAINABILITY SETTINGS
# ============================================
# LIME creates perturbations of the input text and observes
# how predictions change — more samples = more accurate but slower.

LIME_NUM_FEATURES = 10      # Number of top features (words) to highlight
LIME_NUM_SAMPLES = 200      # Number of perturbations to generate

# ============================================
# 7. RANDOM SEED
# ============================================
# Setting a seed ensures reproducible results across runs.

RANDOM_SEED = 42


def print_config():
    """Utility function to display current configuration."""
    print("=" * 50)
    print("  HATE SPEECH DETECTION - CONFIGURATION")
    print("=" * 50)
    print(f"  Model:          {MODEL_NAME}")
    print(f"  Device:         {DEVICE}")
    print(f"  Max Seq Length:  {MAX_SEQ_LENGTH}")
    print(f"  Batch Size:     {BATCH_SIZE}")
    print(f"  Learning Rate:  {LEARNING_RATE}")
    print(f"  Epochs:         {NUM_EPOCHS}")
    print(f"  Labels:         {list(LABEL_TO_ID.keys())}")
    print(f"  Project Root:   {PROJECT_ROOT}")
    print("=" * 50)


# Run this file directly to see the config
if __name__ == "__main__":
    print_config()
