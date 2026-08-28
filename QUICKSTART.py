#!/usr/bin/env python3
"""
QUICK START GUIDE
Sentiment Analysis with PyTorch & DistilBERT
"""

print("""
╔════════════════════════════════════════════════════════════════════╗
║         SENTIMENT ANALYSIS - QUICK START GUIDE                    ║
║              PyTorch + DistilBERT Transfer Learning               ║
╚════════════════════════════════════════════════════════════════════╝

📦 STEP 1: INSTALL DEPENDENCIES
──────────────────────────────────────────────────────────────────────

Option A (Recommended - All at once):
    pip install -r requirements.txt

Option B (Manual):
    pip install torch transformers datasets scikit-learn matplotlib seaborn

Option C (With GPU support):
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    pip install transformers datasets scikit-learn matplotlib seaborn


🚀 STEP 2: TRAIN THE MODEL
──────────────────────────────────────────────────────────────────────

Run the training script:
    python sentiment_analysis_training.py

What it does:
    ✓ Loads 5,000 IMDb movie review samples
    ✓ Splits into train/validation/test sets
    ✓ Tokenizes text using DistilBERT tokenizer (max 128 tokens)
    ✓ Fine-tunes pretrained DistilBERT model for 3 epochs
    ✓ Evaluates on test set with detailed metrics
    ✓ Generates visualizations (loss, accuracy, confusion matrix, ROC)
    ✓ Saves trained model and tokenizer

Expected Results:
    • Accuracy: ~89-91%
    • F1-Score: ~90%
    • Training time: 5-10 min (GPU) or 30-45 min (CPU)

Output Files:
    • sentiment_model.pt          ← Trained model weights
    • tokenizer/                  ← Tokenizer files
    • model_config.json           ← Model metadata
    • training_results.png        ← Visualizations


💡 STEP 3: MAKE PREDICTIONS
──────────────────────────────────────────────────────────────────────

Option A: Run inference script directly:
    python inference.py

This shows example predictions and how to use the model.

Option B: Use in your Python code:

    from inference import predict_sentiment

    # Single prediction
    result = predict_sentiment("This movie was amazing!")
    print(result['sentiment'])    # 'Positive'
    print(result['confidence'])   # 0.9876

    # With probabilities
    result = predict_sentiment("Bad movie", return_probabilities=True)
    print(result['probabilities'])
    # {'negative': 0.95, 'positive': 0.05}

    # Batch predictions
    from inference import predict_batch
    texts = ["Great!", "Terrible", "Ok"]
    results = predict_batch(texts)


🎯 EXAMPLE USAGE
──────────────────────────────────────────────────────────────────────

import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

# Load model and tokenizer
model = SentimentClassifier('distilbert-base-uncased')
model.load_state_dict(torch.load('sentiment_model.pt'))
model.eval()

tokenizer = AutoTokenizer.from_pretrained('tokenizer')

# Make prediction
text = "I absolutely loved this!"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

with torch.no_grad():
    logits = model(**inputs)
    prob = torch.softmax(logits, dim=1)
    sentiment = 'Positive' if prob[0, 1] > prob[0, 0] else 'Negative'
    confidence = prob[0, 1].item() if sentiment == 'Positive' else prob[0, 0].item()

print(f"Sentiment: {sentiment}, Confidence: {confidence:.4f}")


⚙️  CUSTOMIZATION
──────────────────────────────────────────────────────────────────────

Edit CONFIG in sentiment_analysis_training.py to customize:

    CONFIG = {
        'model_name': 'distilbert-base-uncased',  # Change model
        'max_length': 128,                         # Token limit
        'batch_size': 32,                          # Batch size
        'epochs': 3,                               # Number of epochs
        'learning_rate': 2e-5,                     # Learning rate
    }

For better accuracy:
    • Increase epochs: 5-10 instead of 3
    • Use larger model: 'bert-base-uncased' instead of distilbert
    • More training data: increase train_size to 10000+


📊 MODEL DETAILS
──────────────────────────────────────────────────────────────────────

Base Model: DistilBERT (pretrained on Wikipedia + BookCorpus)
    • 6 transformer layers
    • 12 attention heads
    • 768 hidden dimensions
    • ~67M parameters (40% smaller than BERT)

Architecture:
    Input Text
        ↓
    Tokenization (padding/truncation to 128 tokens)
        ↓
    DistilBERT Encoder (frozen, uses pretrained weights)
        ↓
    [CLS] Token Representation (768-dim vector)
        ↓
    Dropout (0.3) + Linear (768→2)
        ↓
    Softmax → Sentiment Label

Training Strategy:
    ✓ Transfer Learning: Leverage pretrained language understanding
    ✓ Fine-tuning: Small learning rate (2e-5) to avoid catastrophic forgetting
    ✓ Frozen encoder: Keep pretrained weights, only train classification head


🔍 TROUBLESHOOTING
──────────────────────────────────────────────────────────────────────

Problem: "ModuleNotFoundError: No module named 'torch'"
Solution: pip install torch transformers datasets scikit-learn matplotlib

Problem: "CUDA out of memory"
Solution: Reduce batch_size to 16 or use distilroberta-base model

Problem: "Model not found"
Solution: Check that sentiment_model.pt and tokenizer/ are in same directory

Problem: "Slow training"
Solution: Use GPU (automatic) or use distilroberta-base instead


📚 LEARN MORE
──────────────────────────────────────────────────────────────────────

Hugging Face Course: https://huggingface.co/course/
DistilBERT Paper: https://arxiv.org/abs/1910.01108
PyTorch Docs: https://pytorch.org/docs/


✨ THAT'S IT!
──────────────────────────────────────────────────────────────────────

1. Install dependencies → pip install -r requirements.txt
2. Train model → python sentiment_analysis_training.py
3. Make predictions → python inference.py

Questions? Check README.md for detailed documentation.

Happy learning! 🚀

""")
