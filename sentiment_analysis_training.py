"""
Sentiment Analysis Text Classifier
Using PyTorch + Hugging Face Transformers (DistilBERT)
Transfer Learning Approach
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Configuration
CONFIG = {
    'model_name': 'distilbert-base-uncased',
    'max_length': 128,
    'batch_size': 32,
    'epochs': 3,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_path': '/mnt/user-data/outputs/sentiment_model.pt',
    'tokenizer_save_path': '/mnt/user-data/outputs/tokenizer_config.json'
}

print(f"Device: {CONFIG['device']}")
print(f"Using model: {CONFIG['model_name']}\n")

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("Loading IMDb dataset...")
dataset = load_dataset('imdb')

# Use subset for faster training while maintaining good performance
train_size = 5000  # Reduced from full 25000 for faster demo
test_size = 1000

train_data = dataset['train'].shuffle(seed=42).select(range(train_size))
test_data = dataset['test'].shuffle(seed=42).select(range(test_size))

# Split training data into train/val
texts_train = train_data['text']
labels_train = train_data['label']

texts_test = test_data['text']
labels_test = test_data['label']

# Further split train into train/val
texts_train, texts_val, labels_train, labels_val = train_test_split(
    texts_train, labels_train,
    test_size=0.2,
    random_state=42,
    stratify=labels_train
)

print(f"Training samples: {len(texts_train)}")
print(f"Validation samples: {len(texts_val)}")
print(f"Test samples: {len(texts_test)}")
print(f"Class distribution - Train: {np.bincount(labels_train)}")
print()

# ============================================================================
# 2. TOKENIZATION & DATASET CREATION
# ============================================================================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# Create datasets
train_dataset = SentimentDataset(texts_train, labels_train, tokenizer, CONFIG['max_length'])
val_dataset = SentimentDataset(texts_val, labels_val, tokenizer, CONFIG['max_length'])
test_dataset = SentimentDataset(texts_test, labels_test, tokenizer, CONFIG['max_length'])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])

print(f"Dataloader batches - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}\n")

# ============================================================================
# 3. MODEL ARCHITECTURE
# ============================================================================
class SentimentClassifier(nn.Module):
    def __init__(self, model_name, num_classes=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Freeze encoder initially (transfer learning)
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
        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        logits = self.classifier(self.dropout(cls_output))
        return logits

# Initialize model
print("Initializing model...")
model = SentimentClassifier(CONFIG['model_name'])
model = model.to(CONFIG['device'])

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}\n")

# ============================================================================
# 4. TRAINING SETUP
# ============================================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], 
                        weight_decay=CONFIG['weight_decay'])

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=1, verbose=True
)

# ============================================================================
# 5. TRAINING & EVALUATION FUNCTIONS
# ============================================================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, all_preds, all_labels

# ============================================================================
# 6. TRAINING LOOP
# ============================================================================
print("Starting training...\n")

train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(CONFIG['epochs']):
    print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
    
    # Training
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, CONFIG['device'])
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Validation
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, CONFIG['device'])
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}\n")

print("Training complete!\n")

# ============================================================================
# 7. EVALUATION ON TEST SET
# ============================================================================
print("Evaluating on test set...")
test_loss, test_acc, test_preds, test_labels = evaluate(
    model, test_loader, criterion, CONFIG['device']
)

# Detailed metrics
precision = precision_score(test_labels, test_preds)
recall = recall_score(test_labels, test_preds)
f1 = f1_score(test_labels, test_preds)
auc = roc_auc_score(test_labels, test_preds)

print(f"\nTest Set Results:")
print(f"  Accuracy:  {test_acc:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  AUC-ROC:   {auc:.4f}\n")

print("Classification Report:")
print(classification_report(test_labels, test_preds, 
      target_names=['Negative', 'Positive']))

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================
print("Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss curves
axes[0, 0].plot(range(1, CONFIG['epochs']+1), train_losses, 'o-', label='Train', linewidth=2)
axes[0, 0].plot(range(1, CONFIG['epochs']+1), val_losses, 's-', label='Validation', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training & Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy curves
axes[0, 1].plot(range(1, CONFIG['epochs']+1), train_accs, 'o-', label='Train', linewidth=2)
axes[0, 1].plot(range(1, CONFIG['epochs']+1), val_accs, 's-', label='Validation', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Training & Validation Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Confusion matrix
cm = confusion_matrix(test_labels, test_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
axes[1, 0].set_title('Confusion Matrix (Test Set)')
axes[1, 0].set_ylabel('True Label')
axes[1, 0].set_xlabel('Predicted Label')

# ROC curve
fpr, tpr, _ = roc_curve(test_labels, test_preds)
axes[1, 1].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.4f}')
axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].set_title('ROC Curve (Test Set)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/training_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: training_results.png\n")

# ============================================================================
# 9. SAVE MODEL & TOKENIZER
# ============================================================================
print("Saving model and tokenizer...")
torch.save(model.state_dict(), CONFIG['save_path'])
tokenizer.save_pretrained('/mnt/user-data/outputs/tokenizer')
print(f"✓ Saved model to: {CONFIG['save_path']}")
print(f"✓ Saved tokenizer to: /mnt/user-data/outputs/tokenizer")

# Save config
import json
config_save = {
    'model_name': CONFIG['model_name'],
    'max_length': CONFIG['max_length'],
    'num_classes': 2,
    'test_accuracy': float(test_acc),
    'test_f1': float(f1),
    'epochs_trained': CONFIG['epochs']
}
with open('/mnt/user-data/outputs/model_config.json', 'w') as f:
    json.dump(config_save, f, indent=2)
print("✓ Saved config to: model_config.json\n")

# ============================================================================
# 10. SUMMARY
# ============================================================================
print("="*60)
print("TRAINING SUMMARY")
print("="*60)
print(f"Dataset: IMDb Movie Reviews")
print(f"Model: DistilBERT (pretrained)")
print(f"Training Strategy: Transfer Learning")
print(f"Epochs: {CONFIG['epochs']}")
print(f"Batch Size: {CONFIG['batch_size']}")
print(f"Learning Rate: {CONFIG['learning_rate']}")
print(f"\nFinal Results on Test Set:")
print(f"  • Accuracy:  {test_acc:.4f}")
print(f"  • Precision: {precision:.4f}")
print(f"  • Recall:    {recall:.4f}")
print(f"  • F1-Score:  {f1:.4f}")
print(f"  • AUC-ROC:   {auc:.4f}")
print("="*60)
