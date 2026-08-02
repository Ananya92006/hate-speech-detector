# Explainable Multilingual Hate Speech Detection
### Using Transformer Models (mBERT + LIME)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Transformers](https://img.shields.io/badge/🤗_Transformers-4.30+-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b.svg)

---

## 📋 Overview

A production-ready system that detects **hate speech** in **multilingual and code-mixed text** (Hindi + English / Hinglish) using Transformer models, with transparent AI explanations.

### Key Features
- 🌐 **Multilingual Support**: Handles Hindi, English, and Hinglish (code-mixed) text
- 🧠 **Transformer-based**: Uses Multilingual BERT (mBERT) for superior accuracy
- 🔍 **Explainable AI**: LIME integration shows *why* the model made each decision
- 😀 **Emoji Understanding**: Converts emojis to meaningful text (😂 → "laugh")
- 🗣️ **Slang Normalization**: 140+ Hinglish slang terms normalized
- 🎯 **3-Class Detection**: Neutral | Offensive | Hate Speech
- 🖥️ **Web Interface**: Beautiful Streamlit app with real-time analysis

### Why This Project is Better Than Traditional ML

| Feature | Traditional ML (SVM/NB) | This Project (mBERT) |
|---------|------------------------|---------------------|
| Multilingual | ❌ Manual feature engineering | ✅ Built-in (104 languages) |
| Context Understanding | ❌ Bag-of-words | ✅ Deep contextual embeddings |
| Code-Mixed Text | ❌ Poor handling | ✅ Native support |
| Training Data Needed | ❌ Lots of labeled data | ✅ Fine-tuning needs less |
| Explainability | ❌ Limited | ✅ LIME word-level explanations |

---

## 🏗️ Project Structure

```
hate-speech-detector/
├── data/
│   └── sample_data.csv         # Hinglish dataset (180+ samples)
├── models/
│   └── saved_model/            # Fine-tuned mBERT checkpoint
├── preprocessing/
│   ├── cleaner.py              # Full cleaning pipeline
│   ├── emoji_handler.py        # Emoji → text conversion
│   └── slang_normalizer.py     # 140+ slang dictionary
├── training/
│   ├── dataset.py              # PyTorch Dataset
│   ├── train.py                # Training loop
│   └── evaluate.py             # Metrics & confusion matrix
├── explainability/
│   └── lime_explainer.py       # LIME integration
├── app/
│   └── streamlit_app.py        # Web interface
├── utils/
│   ├── config.py               # Central configuration
│   └── predictor.py            # Inference pipeline
├── tests/
│   └── test_pipeline.py        # Test suite
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone/navigate to the project
cd hate-speech-detector

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python training/train.py
```

This will:
- Load and preprocess the dataset
- Download mBERT (~700MB, first time only)
- Fine-tune for 3 epochs
- Save the best model to `models/saved_model/`

**Expected output:**
```
  Epoch 1/3: train_loss=0.95, val_acc=0.65
  Epoch 2/3: train_loss=0.45, val_acc=0.82
  Epoch 3/3: train_loss=0.20, val_acc=0.88
  ✅ Best model saved!
```

### 3. Run Tests

```bash
python tests/test_pipeline.py
```

### 4. Launch Web App

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

---

## 📊 Model Details

### Architecture
- **Base Model**: `bert-base-multilingual-cased` (110M parameters)
- **Classification Head**: Linear layer (768 → 3 classes)
- **Tokenizer**: WordPiece (supports 104 languages)

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Epochs | 3 |
| Max Sequence Length | 128 |
| Optimizer | AdamW |
| Loss Function | CrossEntropyLoss |
| Weight Decay | 0.01 |
| Warmup Steps | 100 |

### Classes
| Label | ID | Description |
|-------|-----|-------------|
| 🟢 Neutral | 0 | Normal, non-harmful content |
| 🟡 Offensive | 1 | Rude, insulting but not targeted |
| 🔴 Hate Speech | 2 | Targeted hatred against groups |

---

## 🔧 Preprocessing Pipeline

The text goes through 8 cleaning steps in order:

```
Original:  "@hater yeh bakwas hai 😂😡 #HateSpeech http://spam.com"
    ↓ Lowercase
    ↓ Remove URLs           → removes http://spam.com
    ↓ Remove Mentions       → removes @hater
    ↓ Clean Hashtags        → #HateSpeech → HateSpeech
    ↓ Convert Emojis        → 😂 → "laugh", 😡 → "anger"
    ↓ Normalize Slang       → bakwas → "nonsense"
    ↓ Remove Special Chars
    ↓ Clean Whitespace
Final:     "yeh nonsense hai laugh anger hatespeech"
```

---

## 🔍 LIME Explainability

LIME explains predictions by identifying which words are most important:

**How it works:**
1. Remove random words from the text (perturbation)
2. Run the perturbed text through the model
3. Observe how predictions change
4. Words that change predictions most = most important

**Example:**
```
Input: "in logon ko maaro saale"
Prediction: 🔴 Hate Speech (95.2%)

Word Importance:
  maaro:    +0.32 ⬆️ (strongly supports hate speech)
  saale:    +0.18 ⬆️ (supports)
  logon:    +0.08 ⬆️ (supports)
  ko:       -0.02 ⬇️ (slightly opposes)
  in:       -0.01 ⬇️ (neutral)
```

---

## 📈 Step 10: Optimization & Improvements

### Hyperparameter Tuning
- **Learning Rate**: Try `1e-5` to `5e-5` (use learning rate finder)
- **Batch Size**: Larger (32) for smoother training, smaller (8) for better generalization
- **Epochs**: Monitor validation loss — stop when it increases (early stopping)
- **Max Sequence Length**: 64 for tweets, 128 for longer text, 256 for paragraphs

### Handling Class Imbalance
```python
# Option 1: Weighted Loss Function
class_weights = torch.tensor([1.0, 2.0, 3.0])  # Higher weight for rare classes
loss_fn = CrossEntropyLoss(weight=class_weights.to(device))

# Option 2: Oversampling (duplicate minority class samples)
from imblearn.over_sampling import RandomOverSampler

# Option 3: Focal Loss (reduces weight for easy examples)
```

### IndicBERT vs mBERT
| Feature | mBERT | IndicBERT |
|---------|-------|-----------|
| Languages | 104 | 12 Indian |
| Hindi quality | Good | Better |
| Code-mixed | Good | Better for Indian |
| Model size | 110M | 33M |
| Speed | Moderate | Faster |

**Recommendation**: Use IndicBERT (`ai4bharat/indic-bert`) if your text is primarily Indian languages. Use mBERT if you need broader language coverage.

### Better Slang Dictionary
- Scrape Urban Dictionary for latest slang
- Use community-maintained lists
- Add transliterated Hindi abusive words
- Consider phonetic similarity matching

---

## 🌐 Step 11: Deployment

### Option A: Streamlit Cloud (Easiest)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file: `app/streamlit_app.py`
5. Click Deploy!

**Note:** Add a `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 1024
```

### Option B: HuggingFace Spaces
1. Create account at [huggingface.co](https://huggingface.co)
2. Create a new Space (Streamlit SDK)
3. Upload all project files
4. Add `requirements.txt`
5. The app auto-deploys!

### Option C: Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501"]
```

---

## 🧪 Test Cases

| # | Input | Expected Class |
|---|-------|---------------|
| 1 | "yeh bakwas hai 😂" | Offensive |
| 2 | "tum pagal ho kya" | Offensive |
| 3 | "aaj mausam bahut accha hai" | Neutral |
| 4 | "happy birthday bhai 🎉🥳" | Neutral |
| 5 | "in logon ko maaro saale 😡🔪" | Hate Speech |
| 6 | "yeh log insaan nahi hain" | Hate Speech |
| 7 | "chapri log hain sab" | Offensive |
| 8 | "coding sikhte raho, future bright hai 💻" | Neutral |
| 9 | "inki puri community ko ban karo" | Hate Speech |

---

## 📚 References

1. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2019)
2. Ribeiro et al., "Why Should I Trust You?: LIME" (2016)
3. HASOC - Hate Speech and Offensive Content Identification
4. Bohra et al., "Hindi-English Code-Mixed Social Media Text" (2018)

---

## 📄 License

This project is for educational/academic purposes. The dataset contains offensive content for research purposes only.
