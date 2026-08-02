"""
==========================================================
Model Training Script
==========================================================

PURPOSE:
    Fine-tune a pretrained Multilingual BERT (mBERT) model
    for hate speech classification (3 classes).

WHAT IS FINE-TUNING?
    Instead of training from scratch (which needs millions of
    samples), we take a model that already understands language
    (mBERT was trained on 104 languages!) and teach it our
    specific task — hate speech detection.

    Think of it like this:
    - mBERT already knows grammar, meaning, context
    - We just teach it: "these patterns = hate speech"
    - This is why we need only ~500 samples instead of millions!

TRAINING PROCESS:
    1. Load pretrained mBERT
    2. Add a classification head (3 output neurons)
    3. Feed our data through the model
    4. Calculate loss (how wrong the predictions are)
    5. Backpropagate (adjust model weights to reduce loss)
    6. Repeat for N epochs
    7. Save the best model
"""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    MODEL_NAME, NUM_LABELS, MAX_SEQ_LENGTH, MODEL_DIR,
    LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, WARMUP_STEPS,
    WEIGHT_DECAY, TRAIN_SPLIT, DEVICE, RANDOM_SEED,
    SAMPLE_DATA_PATH, ID_TO_LABEL, EARLY_STOPPING_PATIENCE
)
from preprocessing.cleaner import TextCleaner
from training.dataset import create_data_loaders


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    WHY: Without setting seeds, results would be different
    every run due to random initialization and shuffling.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_preprocess_data(data_path):
    """
    Load dataset from CSV and apply preprocessing.
    
    Steps:
    1. Read CSV file with pandas
    2. Show dataset statistics
    3. Apply text cleaning pipeline
    4. Return cleaned texts and labels
    
    Args:
        data_path (str): Path to the CSV file
        
    Returns:
        tuple: (texts_list, labels_list, dataframe)
    """
    print("\n" + "=" * 60)
    print("  STEP 1: LOADING DATASET")
    print("=" * 60)
    
    # Load the CSV
    df = pd.read_csv(data_path)
    
    # Display dataset info
    print(f"\n  Total samples:    {len(df)}")
    print(f"  Columns:          {list(df.columns)}")
    print(f"\n  Class distribution:")
    for label_id, count in df['label'].value_counts().sort_index().items():
        label_name = ID_TO_LABEL[label_id]
        print(f"    {label_name} ({label_id}): {count} samples ({count/len(df)*100:.1f}%)")
    
    # Show sample rows
    print(f"\n  Sample rows:")
    for i in range(min(3, len(df))):
        text = df.iloc[i]['text'][:60]
        label = ID_TO_LABEL[df.iloc[i]['label']]
        print(f"    [{label:>12}] {text}...")
    
    # ============================================
    # Apply preprocessing
    # ============================================
    print("\n" + "=" * 60)
    print("  STEP 2: PREPROCESSING TEXT")
    print("=" * 60)
    
    cleaner = TextCleaner()
    
    # Clean all texts
    print("  Cleaning texts...")
    df['cleaned_text'] = df['text'].apply(cleaner.clean)
    
    # Show before vs after for a few samples
    print("\n  Before -> After examples:")
    for i in range(min(5, len(df))):
        before = df.iloc[i]['text'][:50]
        after = df.iloc[i]['cleaned_text'][:50]
        print(f"    BEFORE: {before}")
        print(f"    AFTER:  {after}")
        print()
    
    # Remove any empty texts after cleaning
    df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)
    
    texts = df['cleaned_text'].tolist()
    labels = df['label'].tolist()
    
    print(f"  Final dataset size: {len(texts)} samples")
    
    return texts, labels, df


def train_model(data_path=None):
    """
    Main training function — orchestrates the entire training process.
    
    This function:
    1. Loads and preprocesses data
    2. Initializes the model and tokenizer
    3. Creates data loaders
    4. Runs the training loop
    5. Evaluates on validation set
    6. Saves the best model
    
    Args:
        data_path (str): Path to dataset CSV. Defaults to sample_data.csv
        
    Returns:
        tuple: (model, tokenizer, training_history)
    """
    if data_path is None:
        data_path = SAMPLE_DATA_PATH
    
    # Set reproducibility seed
    set_seed(RANDOM_SEED)
    
    print("\n" + "=" * 60)
    print("  HATE SPEECH DETECTION - MODEL TRAINING")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Model:  {MODEL_NAME}")
    
    # ============================================
    # 1. Load and preprocess data
    # ============================================
    texts, labels, df = load_and_preprocess_data(data_path)
    
    # ============================================
    # 2. Load mBERT tokenizer and model
    # ============================================
    print("\n" + "=" * 60)
    print("  STEP 3: LOADING mBERT MODEL")
    print("=" * 60)
    
    print(f"  Loading tokenizer: {MODEL_NAME}...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    print(f"  Loading model: {MODEL_NAME}...")
    print(f"  Adding classification head with {NUM_LABELS} labels...")
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS  # 3 classes: Neutral, Offensive, Hate Speech
    )
    
    # Move model to GPU/CPU
    model = model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # ============================================
    # 3. Create DataLoaders
    # ============================================
    print("\n" + "=" * 60)
    print("  STEP 4: CREATING DATA LOADERS")
    print("=" * 60)
    
    train_loader, val_loader, train_dataset, val_dataset = create_data_loaders(
        texts, labels, tokenizer,
        max_length=MAX_SEQ_LENGTH,
        batch_size=BATCH_SIZE,
        train_split=TRAIN_SPLIT
    )
    
    print(f"  Training batches:   {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # ============================================
    # 4. Setup optimizer and scheduler
    # ============================================
    print("\n" + "=" * 60)
    print("  STEP 5: SETTING UP OPTIMIZER")
    print("=" * 60)
    
    # AdamW optimizer — the standard for fine-tuning transformers
    # It's like regular Adam but with proper weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Total training steps = batches × epochs
    total_steps = len(train_loader) * NUM_EPOCHS
    
    # Learning rate scheduler: starts high, gradually decreases
    # Warm-up: gradually increases LR for the first few steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(WARMUP_STEPS, total_steps // 4),
        num_training_steps=total_steps
    )
    
    # Loss function: CrossEntropyLoss for multi-class classification
    loss_fn = CrossEntropyLoss()
    
    print(f"  Optimizer:       AdamW (lr={LEARNING_RATE})")
    print(f"  Loss function:   CrossEntropyLoss")
    print(f"  LR Scheduler:    Linear with warmup")
    print(f"  Total steps:     {total_steps}")
    
    # ============================================
    # 5. TRAINING LOOP
    # ============================================
    print("\n" + "=" * 60)
    print("  STEP 6: TRAINING")
    print("=" * 60)
    
    # Track training history for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    best_val_accuracy = 0
    best_epoch = 0
    patience_counter = 0  # For early stopping
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        # ----- Training Phase -----
        model.train()  # Set model to training mode (enables dropout)
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        
        # Progress bar for training
        train_bar = tqdm(
            train_loader, 
            desc=f"  Epoch {epoch+1}/{NUM_EPOCHS} [Train]",
            leave=True
        )
        
        for batch in train_bar:
            # Move batch to device (GPU/CPU)
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels_batch = batch['labels'].to(DEVICE)
            
            # Zero the gradients from previous step
            optimizer.zero_grad()
            
            # Forward pass: feed data through the model
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_batch
            )
            
            # Get loss (model computes it internally when labels are provided)
            loss = outputs.loss
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Gradient clipping: prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update model weights
            optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Track metrics
            total_train_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            correct_train += (predictions == labels_batch).sum().item()
            total_train += labels_batch.size(0)
            
            # Update progress bar
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct_train/total_train:.3f}'
            })
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        
        # ----- Validation Phase -----
        model.eval()  # Set model to evaluation mode (disables dropout)
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        
        val_bar = tqdm(
            val_loader,
            desc=f"  Epoch {epoch+1}/{NUM_EPOCHS} [Val]  ",
            leave=True
        )
        
        with torch.no_grad():  # Disable gradient computation (saves memory)
            for batch in val_bar:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels_batch = batch['labels'].to(DEVICE)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_batch
                )
                
                total_val_loss += outputs.loss.item()
                predictions = torch.argmax(outputs.logits, dim=1)
                correct_val += (predictions == labels_batch).sum().item()
                total_val += labels_batch.size(0)
                
                val_bar.set_postfix({
                    'loss': f'{outputs.loss.item():.4f}',
                    'acc': f'{correct_val/total_val:.3f}'
                })
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val
        
        epoch_time = time.time() - epoch_start
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # Print epoch summary
        print(f"\n  Epoch {epoch+1}/{NUM_EPOCHS} Summary:")
        print(f"    Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
        print(f"    Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_accuracy:.4f}")
        print(f"    Time:       {epoch_time:.1f}s")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            
            # Create model directory
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            
            print(f"    [BEST] Model saved! (accuracy: {val_accuracy:.4f})")
            patience_counter = 0  # Reset patience
        else:
            patience_counter += 1
            print(f"    No improvement for {patience_counter}/{EARLY_STOPPING_PATIENCE} epochs")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n  Early stopping triggered after {epoch+1} epochs!")
                break

    
    # ============================================
    # 6. Training Complete
    # ============================================
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Best epoch:        {best_epoch}")
    print(f"  Best val accuracy: {best_val_accuracy:.4f}")
    print(f"  Model saved to:    {MODEL_DIR}")
    
    return model, tokenizer, history


# ============================================
# RUN TRAINING
# ============================================
if __name__ == "__main__":
    model, tokenizer, history = train_model()
    
    print("\n  Training History:")
    for epoch, (tl, vl, va) in enumerate(zip(
        history['train_loss'], history['val_loss'], history['val_accuracy']
    )):
        print(f"    Epoch {epoch+1}: train_loss={tl:.4f}, val_loss={vl:.4f}, val_acc={va:.4f}")
