"""
==========================================================
Predictor Module — Inference Pipeline
==========================================================

PURPOSE:
    Provides a simple, clean interface for making predictions:
    
        predictor = HateSpeechPredictor()
        result = predictor.predict("yeh bakwas hai 😂")
        # result = {'label': 'Offensive', 'confidence': 0.87, ...}

HOW IT WORKS:
    1. Load the saved model and tokenizer
    2. Preprocess the input text (clean, normalize)
    3. Tokenize and feed through the model
    4. Apply softmax to get probabilities
    5. Return the predicted label and confidence

    This is the bridge between the trained model and the UI.
"""

import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import BertTokenizer, BertForSequenceClassification
from preprocessing.cleaner import TextCleaner
from utils.config import (
    MODEL_DIR, MODEL_NAME, MAX_SEQ_LENGTH, 
    DEVICE, ID_TO_LABEL, NUM_LABELS,
    LABEL_COLORS, LABEL_EMOJIS
)


class HateSpeechPredictor:
    """
    End-to-end prediction pipeline for hate speech detection.
    
    This class encapsulates:
    - Model loading
    - Text preprocessing
    - Tokenization
    - Inference
    - Result formatting
    
    Usage:
        predictor = HateSpeechPredictor()
        result = predictor.predict("some text")
        print(result['label'], result['confidence'])
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the predictor by loading model and tokenizer.
        
        Args:
            model_path (str): Path to saved model directory.
                            If None, uses the default MODEL_DIR.
        """
        if model_path is None:
            model_path = MODEL_DIR
        
        self.device = DEVICE
        self.cleaner = TextCleaner()
        
        # Load model and tokenizer
        if os.path.exists(model_path):
            print(f"  Loading model from: {model_path}")
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertForSequenceClassification.from_pretrained(model_path)
        else:
            # If no fine-tuned model, load base mBERT (for testing)
            print(f"  [WARNING] No fine-tuned model found at {model_path}")
            print(f"  Loading base model: {MODEL_NAME}")
            self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
            self.model = BertForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=NUM_LABELS
            )
        
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        print(f"  [OK] Model loaded on {self.device}")
    
    def predict(self, text, preprocess=True):
        """
        Predict the class of a single text input.
        
        Args:
            text (str): Input text (can be raw or preprocessed)
            preprocess (bool): Whether to apply preprocessing
            
        Returns:
            dict: {
                'original_text': str,       # Original input
                'cleaned_text': str,        # After preprocessing
                'label': str,               # Predicted label name
                'label_id': int,            # Numeric label
                'confidence': float,        # Confidence (0-1)
                'probabilities': dict,      # Per-class probabilities
                'emoji': str,               # Label emoji
                'color': str                # Label color code
            }
        """
        # Step 1: Preprocess
        original_text = text
        if preprocess:
            cleaned_text = self.cleaner.clean(text)
        else:
            cleaned_text = text
        
        # Step 2: Tokenize
        encoding = self.tokenizer(
            cleaned_text,
            add_special_tokens=True,
            max_length=MAX_SEQ_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Step 3: Get predictions
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Step 4: Apply softmax to get probabilities
        probabilities = torch.softmax(outputs.logits, dim=1)
        probs_np = probabilities.cpu().numpy()[0]
        
        # Step 5: Get the predicted class
        predicted_id = int(np.argmax(probs_np))
        predicted_label = ID_TO_LABEL[predicted_id]
        confidence = float(probs_np[predicted_id])
        
        # Build per-class probability dict
        prob_dict = {
            ID_TO_LABEL[i]: float(probs_np[i]) 
            for i in range(len(probs_np))
        }
        
        return {
            'original_text': original_text,
            'cleaned_text': cleaned_text,
            'label': predicted_label,
            'label_id': predicted_id,
            'confidence': confidence,
            'probabilities': prob_dict,
            'emoji': LABEL_EMOJIS.get(predicted_label, ""),
            'color': LABEL_COLORS.get(predicted_label, "#ffffff")
        }
    
    def predict_batch(self, texts, preprocess=True):
        """
        Predict classes for multiple texts at once.
        
        Args:
            texts (list[str]): List of input texts
            preprocess (bool): Whether to apply preprocessing
            
        Returns:
            list[dict]: List of prediction results
        """
        return [self.predict(text, preprocess) for text in texts]
    
    def predict_proba(self, texts, batch_size=32):
        """
        Return class probabilities for a list of texts.
        
        This method is specifically designed for LIME compatibility.
        LIME requires: input = list of strings, output = numpy array of probabilities.
        
        Uses batched inference for much faster processing — critical
        for LIME which sends hundreds of perturbed texts at once.
        
        Args:
            texts (list[str]): List of text strings
            batch_size (int): Number of texts to process at once
            
        Returns:
            np.array: Shape (n_samples, n_classes) probability array
        """
        all_probs = []
        
        # Process texts in batches for speed
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize entire batch at once
            encoding = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=MAX_SEQ_LENGTH,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            probs = torch.softmax(outputs.logits, dim=1)
            all_probs.append(probs.cpu().numpy())
        
        return np.concatenate(all_probs, axis=0)


# ============================================
# DEMONSTRATION
# ============================================
if __name__ == "__main__":
    predictor = HateSpeechPredictor()
    
    test_texts = [
        "yeh bakwas hai 😂",
        "tum pagal ho kya",
        "aaj mausam bahut accha hai",
        "in logon ko maaro saale",
        "happy birthday bhai 🎉🥳",
    ]
    
    print("\n" + "=" * 60)
    print("  PREDICTION DEMO")
    print("=" * 60)
    
    for text in test_texts:
        result = predictor.predict(text)
        print(f"\n  Input:      {text}")
        print(f"  Cleaned:    {result['cleaned_text']}")
        print(f"  Prediction: {result['emoji']} {result['label']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  All probs:  {result['probabilities']}")
        print("-" * 60)
