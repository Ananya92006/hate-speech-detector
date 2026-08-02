"""
==========================================================
PyTorch Dataset for Hate Speech Detection
==========================================================

PURPOSE:
    Creates a custom PyTorch Dataset that:
    1. Takes raw text data and labels
    2. Tokenizes text using the mBERT tokenizer
    3. Returns tensors ready for the model

WHY A CUSTOM DATASET?
    PyTorch's DataLoader needs a Dataset object to efficiently:
    - Batch data together
    - Shuffle data each epoch
    - Load data in parallel (num_workers)
    
    This is much more efficient than feeding data one-by-one!

KEY CONCEPT:
    __getitem__() is called for each sample. It returns:
    - input_ids: Token IDs (words → numbers)
    - attention_mask: Which tokens are real vs padding (1 vs 0)
    - label: The ground truth class (0, 1, or 2)
"""

import torch
import random
from torch.utils.data import Dataset
from transformers import BertTokenizer


class HateSpeechDataset(Dataset):
    """
    Custom PyTorch Dataset for hate speech text classification.
    
    What this does:
    1. Stores texts and labels
    2. When indexed (dataset[i]), tokenizes text on-the-fly
    3. Returns tensors that BERT can directly process
    
    Why tokenize on-the-fly?
    - Saves memory (don't store all tokenized text at once)
    - More flexible (can change max_length without re-processing)
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=128, augment=False):
        """
        Initialize the dataset.
        
        Args:
            texts (list[str]): List of preprocessed text strings
            labels (list[int]): List of integer labels (0, 1, or 2)
            tokenizer: Hugging Face tokenizer (mBERT tokenizer)
            max_length (int): Maximum sequence length for tokenization
                             128 is good for social media text (usually short)
            augment (bool): Whether to apply data augmentation (random word dropout)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
    
    def __len__(self):
        """
        Return the total number of samples.
        
        This is required by PyTorch DataLoader to know
        how many batches to create.
        """
        return len(self.texts)
    
    def _augment_text(self, text):
        """
        Apply random word dropout augmentation.
        
        Randomly drops 15% of words from the text, creating
        slightly different versions each time. This helps the
        model learn to not rely on any single word.
        """
        words = text.split()
        if len(words) <= 2:
            return text  # Don't augment very short texts
        
        # Keep each word with 85% probability
        kept_words = [w for w in words if random.random() > 0.15]
        
        # Ensure at least 2 words remain
        if len(kept_words) < 2:
            kept_words = random.sample(words, min(2, len(words)))
        
        return ' '.join(kept_words)
    
    def __getitem__(self, idx):
        """
        Get a single sample by index.
        
        This is called by the DataLoader for each sample.
        
        Process:
        1. Get the raw text and label at position `idx`
        2. Optionally apply augmentation
        3. Tokenize the text using mBERT tokenizer
        4. Return as a dictionary of tensors
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: {
                'input_ids': tensor of token IDs,
                'attention_mask': tensor of attention mask (1=real, 0=pad),
                'labels': tensor with the label
            }
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Apply augmentation if enabled (training only)
        if self.augment:
            text = self._augment_text(text)
        
        # Tokenize the text
        # - padding='max_length': Pad short texts to max_length
        # - truncation=True: Cut long texts to max_length
        # - return_tensors='pt': Return PyTorch tensors
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,      # Add [CLS] and [SEP] tokens
            max_length=self.max_length,
            padding='max_length',          # Pad to max_length
            truncation=True,               # Truncate if too long
            return_attention_mask=True,     # Create attention mask
            return_tensors='pt'            # Return PyTorch tensors
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),          # Shape: (max_length,)
            'attention_mask': encoding['attention_mask'].flatten(), # Shape: (max_length,)
            'labels': torch.tensor(label, dtype=torch.long)        # Scalar tensor
        }


def create_data_loaders(texts, labels, tokenizer, max_length=128, 
                         batch_size=16, train_split=0.8):
    """
    Create train and validation DataLoaders from raw data.
    
    This function:
    1. Splits data into train/validation sets
    2. Creates Dataset objects for each split
    3. Wraps them in DataLoaders for batching
    
    Args:
        texts (list[str]): All text data
        labels (list[int]): All labels
        tokenizer: Hugging Face tokenizer
        max_length (int): Max token sequence length
        batch_size (int): Samples per batch
        train_split (float): Fraction of data for training (0.8 = 80%)
        
    Returns:
        tuple: (train_loader, val_loader, train_dataset, val_dataset)
    """
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split
    
    # Split data into train and validation sets
    # stratify ensures equal class distribution in both sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=1 - train_split,
        random_state=42,        # For reproducibility
        stratify=labels          # Maintain class ratios
    )
    
    print(f"  Training samples:   {len(train_texts)}")
    print(f"  Validation samples: {len(val_texts)}")
    
    # Create Dataset objects
    # augment=True for training to apply random word dropout
    train_dataset = HateSpeechDataset(train_texts, train_labels, tokenizer, max_length, augment=True)
    val_dataset = HateSpeechDataset(val_texts, val_labels, tokenizer, max_length, augment=False)
    
    # Create DataLoaders
    # - shuffle=True for training (randomize order each epoch)
    # - shuffle=False for validation (consistent evaluation)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # Randomize training data
        num_workers=0,          # Use 0 for Windows compatibility
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,          # Don't shuffle validation data
        num_workers=0,
        drop_last=False
    )
    
    return train_loader, val_loader, train_dataset, val_dataset


# ============================================
# DEMONSTRATION
# ============================================
if __name__ == "__main__":
    from transformers import BertTokenizer
    
    # Load the mBERT tokenizer
    print("Loading mBERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    # Sample data
    sample_texts = [
        "yeh bahut accha hai",
        "bakwas content hai yeh",
        "in logon ko maaro",
    ]
    sample_labels = [0, 1, 2]
    
    # Create dataset
    dataset = HateSpeechDataset(sample_texts, sample_labels, tokenizer)
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Get one sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Input IDs shape:      {sample['input_ids'].shape}")
    print(f"  Attention mask shape:  {sample['attention_mask'].shape}")
    print(f"  Label:                 {sample['labels'].item()}")
    print(f"  First 20 token IDs:   {sample['input_ids'][:20].tolist()}")
