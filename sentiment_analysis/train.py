import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time
import os
from tqdm import tqdm
from collections import Counter

from data_processor import TextProcessor, load_and_prepare_data
from model import TextClassifier

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        predictions = torch.argmax(logits, dim=-1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Calculate metrics
    epoch_loss = total_loss / len(dataloader)
    epoch_accuracy = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_accuracy

def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            # Store predictions and labels for metrics
            predictions = torch.argmax(logits, dim=-1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    eval_loss = total_loss / len(dataloader)
    eval_accuracy = accuracy_score(all_labels, all_preds)
    
    target_names = ['Negative', 'Positive', 'Neutral', 'Irrelevant']
    report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return eval_loss, eval_accuracy, report, cm

def create_class_weights(labels):
    """Create class weights for loss function"""
    # Count instances of each class
    counter = Counter(labels)
    class_counts = [counter[i] for i in range(len(counter))]
    
    # Calculate weights (inverse of frequency)
    total_samples = sum(class_counts)
    class_weights = [total_samples / (len(class_counts) * count) for count in class_counts]
    
    # Convert to tensor
    weights = torch.tensor(class_weights, dtype=torch.float)
    return weights

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    DATA_PATH = "data/twitter_training.csv"
    VAL_DATA_PATH = "data/twitter_validation.csv" 
    BATCH_SIZE = 32
    MAX_SEQ_LENGTH = 128
    VOCAB_MIN_FREQ = 2
    D_MODEL = 256
    NUM_HEADS = 8
    NUM_LAYERS = 4
    D_FF = 1024
    DROPOUT = 0.2  # Increased dropout for better regularization
    NUM_CLASSES = 4
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-5  # L2 regularization
    NUM_EPOCHS = 15  # Increased number of epochs
    SAVE_PATH = "models/best_model.pth"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    processor = TextProcessor(max_seq_length=MAX_SEQ_LENGTH, min_freq=VOCAB_MIN_FREQ)
    
    print("Loading and preparing data...")
    train_loader, val_loader, test_loader, train_labels = load_and_prepare_data(
        DATA_PATH, 
        processor, 
        BATCH_SIZE,
        return_train_labels=True,
        use_balanced_sampling=True  # Use balanced sampling when creating the DataLoader
    )
    
    label_counts = Counter(train_labels)
    print("Class distribution in training data:")
    for label_name, label_id in {'Negative': 0, 'Positive': 1, 'Neutral': 2, 'Irrelevant': 3}.items():
        print(f"  {label_name}: {label_counts[label_id]} ({label_counts[label_id]/len(train_labels)*100:.1f}%)")
    
    class_weights = create_class_weights(train_labels)
    print(f"Class weights: {class_weights}")
    class_weights = class_weights.to(device)
    
    # Initialize model
    print("Initializing transformer model...")
    model = TextClassifier(
        vocab_size=len(processor.word2idx),
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LENGTH,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, 
                  float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    print("Starting training...")
    best_val_accuracy = 0
    best_val_f1_macro = 0
    patience = 3  # early stopping patience
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # Training
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Val
        val_loss, val_accuracy, val_report, val_cm = evaluate(model, val_loader, criterion, device)
        
        # Calculate macro F1 score
        val_f1_macro = sum([val_report[cls]['f1-score'] for cls in ['Negative', 'Positive', 'Neutral', 'Irrelevant']]) / 4
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Time: {epoch_time:.2f}s | LR: {current_lr:.6f}")
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | Val Macro F1: {val_f1_macro:.4f}")
        
        print("Confusion Matrix:")
        print(val_cm)
        
        # Save best model based on macro F1 score (better for imbalanced datasets)
        if val_f1_macro > best_val_f1_macro:
            best_val_f1_macro = val_f1_macro
            best_val_accuracy = val_accuracy
            patience_counter = 0
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"New best model saved with validation macro F1: {val_f1_macro:.4f}")
        else:
            patience_counter += 1
            print(f"Validation macro F1 did not improve. Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Final evaluation on test set
    print("\nLoading best model for testing...")
    model.load_state_dict(torch.load(SAVE_PATH))
    test_loss, test_accuracy, test_report, test_cm = evaluate(model, test_loader, criterion, device)
    
    print("\n===== Final Test Results =====")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    print("\nClassification Report:")
    for label in ['Negative', 'Positive', 'Neutral', 'Irrelevant']:
        print(f"\n{label}:")
        print(f"  Precision: {test_report[label]['precision']:.4f}")
        print(f"  Recall: {test_report[label]['recall']:.4f}")
        print(f"  F1-score: {test_report[label]['f1-score']:.4f}")
        
    print("\nConfusion Matrix:")
    print(test_cm)
    
    # Calculate overall macro F1 score
    macro_f1 = sum([test_report[cls]['f1-score'] for cls in ['Negative', 'Positive', 'Neutral', 'Irrelevant']]) / 4
    print(f"\nOverall Macro F1 Score: {macro_f1:.4f}")

if __name__ == "__main__":
    main()