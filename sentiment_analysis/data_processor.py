import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
from typing import List, Dict, Tuple
import numpy as np

class TextProcessor:
    def __init__(self, max_seq_length: int = 128, min_freq: int = 2):
        self.max_seq_length = max_seq_length
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_freq = Counter()
        
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text data"""
        text = text.lower()
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        text = re.sub(r'[^\w\s]', '', text)
        
        text = ' '.join(text.split())
        
        return text
    
    def build_vocabulary(self, texts: List[str]) -> None:
        """Build vocabulary from texts"""
        for text in texts:
            words = self.preprocess_text(str(text)).split()
            self.word_freq.update(words)
        
        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
                
        print(f"Vocabulary size: {len(self.word2idx)}")
    
    def text_to_sequence(self, text: str) -> List[int]:
        """Convert text to sequence of token ids"""
        words = self.preprocess_text(str(text)).split()
        sequence = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        
        if len(sequence) > self.max_seq_length:
            sequence = sequence[:self.max_seq_length]
        else:
            sequence = sequence + [self.word2idx['<PAD>']] * (self.max_seq_length - len(sequence))
            
        return sequence
    
    def create_attention_mask(self, sequence: List[int]) -> List[int]:
        """Create attention mask for padding tokens"""
        return [1 if token != self.word2idx['<PAD>'] else 0 for token in sequence]

class TextClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], processor: TextProcessor):
        self.texts = texts
        self.labels = labels
        self.processor = processor
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        sequence = self.processor.text_to_sequence(text)
        attention_mask = self.processor.create_attention_mask(sequence)
        
        return {
            'input_ids': torch.tensor(sequence, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_balanced_sampler(labels):
    """Create a weighted sampler to address class imbalance"""
    counter = Counter(labels)
    class_counts = [counter[i] for i in range(len(counter))]
    
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = weights[torch.tensor(labels)]
    
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler

def load_and_prepare_data(data_path: str, processor: TextProcessor, batch_size: int = 32, 
                         train_ratio: float = 0.8, val_ratio: float = 0.1, 
                         return_train_labels: bool = False, use_balanced_sampling: bool = True) -> Tuple:
    """Tải dữ liệu và chuẩn bị DataLoaders để training, validation và testing"""
    
    df = pd.read_csv(data_path)
    
    if 'sentiment' in df.columns:
        sentiment_map = {'Negative': 0, 'Positive': 1, 'Neutral': 2, 'Irrelevant': 3}
        df['label'] = df['sentiment'].map(sentiment_map)
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    processor.build_vocabulary(texts)
    
    n = len(texts)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_texts, train_labels = texts[:train_end], labels[:train_end]
    val_texts, val_labels = texts[train_end:val_end], labels[train_end:val_end]
    test_texts, test_labels = texts[val_end:], labels[val_end:]
    
    train_dataset = TextClassificationDataset(train_texts, train_labels, processor)
    val_dataset = TextClassificationDataset(val_texts, val_labels, processor)
    test_dataset = TextClassificationDataset(test_texts, test_labels, processor)
    
    # Create dataloaders
    if use_balanced_sampling:
        train_sampler = create_balanced_sampler(train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    if return_train_labels:
        return train_loader, val_loader, test_loader, train_labels
    else:
        return train_loader, val_loader, test_loader