"""
Sentiment Analysis Inference Script
Load trained model and make predictions on new text
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'model_path': '/mnt/user-data/outputs/sentiment_model.pt',
    'tokenizer_path': '/mnt/user-data/outputs/tokenizer',
    'config_path': '/mnt/user-data/outputs/model_config.json',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ============================================================================
# MODEL ARCHITECTURE (must match training script)
# ============================================================================
class SentimentClassifier(nn.Module):
    def __init__(self, model_name, num_classes=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Freeze encoder (same as training)
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        hidden_size = self.encoder.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_output))
        return logits

# ============================================================================
# LOAD MODEL
# ============================================================================
print("Loading model and tokenizer...")

# Load config
with open(CONFIG['config_path'], 'r') as f:
    model_config = json.load(f)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(CONFIG['tokenizer_path'])

# Load model
model = SentimentClassifier(model_config['model_name'], num_classes=2)
model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
model = model.to(CONFIG['device'])
model.eval()

print(f"✓ Model loaded successfully")
print(f"✓ Model trained for {model_config['epochs_trained']} epochs")
print(f"✓ Test Accuracy: {model_config['test_accuracy']:.4f}")
print(f"✓ Test F1-Score: {model_config['test_f1']:.4f}\n")

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def predict_sentiment(text, return_probabilities=False):
    """
    Predict sentiment for input text
    
    Args:
        text (str): Input text to classify
        return_probabilities (bool): If True, return confidence scores
    
    Returns:
        dict: Prediction results with sentiment label and confidence
    """
    # Tokenize
    encodings = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=model_config['max_length'],
        return_tensors='pt'
    )
    
    input_ids = encodings['input_ids'].to(CONFIG['device'])
    attention_mask = encodings['attention_mask'].to(CONFIG['device'])
    
    # Predict
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][pred_class].item()
    
    # Map to labels
    label_map = {0: 'Negative', 1: 'Positive'}
    
    result = {
        'text': text,
        'sentiment': label_map[pred_class],
        'confidence': confidence,
        'pred_class': pred_class
    }
    
    if return_probabilities:
        result['probabilities'] = {
            'negative': probabilities[0][0].item(),
            'positive': probabilities[0][1].item()
        }
    
    return result

# ============================================================================
# BATCH PREDICTION
# ============================================================================
def predict_batch(texts, return_probabilities=False):
    """
    Predict sentiment for multiple texts
    
    Args:
        texts (list): List of texts to classify
        return_probabilities (bool): If True, return confidence scores
    
    Returns:
        list: List of prediction results
    """
    results = []
    for text in texts:
        result = predict_sentiment(text, return_probabilities)
        results.append(result)
    return results

# ============================================================================
# EXAMPLES & TESTING
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("SENTIMENT ANALYSIS - INFERENCE EXAMPLES")
    print("="*70 + "\n")
    
    # Test examples
    test_examples = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible film. Waste of time and money. I fell asleep halfway through.",
        "It was okay, nothing special but not terrible either.",
        "One of the best movies I've ever seen! Highly recommend!",
        "The worst acting I've ever witnessed. Completely unbearable."
    ]
    
    print("Single Predictions with Probabilities:\n")
    for text in test_examples:
        result = predict_sentiment(text, return_probabilities=True)
        print(f"Text: {text[:60]}...")
        print(f"  └─ Sentiment: {result['sentiment']}")
        print(f"  └─ Confidence: {result['confidence']:.4f}")
        print(f"  └─ Probabilities: Negative={result['probabilities']['negative']:.4f}, " +
              f"Positive={result['probabilities']['positive']:.4f}\n")
    
    print("\n" + "="*70)
    print("HOW TO USE THIS SCRIPT IN YOUR CODE:")
    print("="*70 + "\n")
    
    print("Option 1: Single Prediction")
    print("-" * 70)
    print("""
from inference import predict_sentiment

result = predict_sentiment("This movie was great!")
print(result['sentiment'])  # Output: 'Positive'
print(result['confidence'])  # Output: 0.9823 (example)
    """)
    
    print("\nOption 2: Batch Predictions")
    print("-" * 70)
    print("""
from inference import predict_batch

texts = ["Great movie!", "Really bad", "It was okay"]
results = predict_batch(texts, return_probabilities=True)

for result in results:
    print(f"{result['text']}: {result['sentiment']} ({result['confidence']:.4f})")
    """)
    
    print("\nOption 3: Custom Text Input")
    print("-" * 70)
    print("""
# For interactive use:
user_text = input("Enter text to classify: ")
result = predict_sentiment(user_text, return_probabilities=True)

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Probabilities: {result['probabilities']}")
    """)

print("\n" + "="*70)
print("MODEL INFORMATION")
print("="*70)
print(f"Base Model: {model_config['model_name']}")
print(f"Max Token Length: {model_config['max_length']}")
print(f"Device: {CONFIG['device'].upper()}")
print(f"Model Parameters: ~67M (DistilBERT)")
print("="*70)
