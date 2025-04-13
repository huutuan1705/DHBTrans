import random
import torch
import torchvision
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset, Subset
from utils.transform import get_transform

class TransformSubset(Subset):
    """Custom Subset class that applies transforms to the dataset after subsetting"""
    def __init__(self, dataset, indices, transform=None):
        super(TransformSubset, self).__init__(dataset, indices)
        self.transform = transform
    
    def __getitem__(self, idx):
        image, label = super(TransformSubset, self).__getitem__(idx)
        if self.transform:
            image = self.transform(image)
        return image, label
    
class CiFar10_Dataset(Dataset):
    def __init__(self, args, mode, query_size=1000, train_size=5000):
        super(CiFar10_Dataset, self).__init__()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.mode = mode
        
        self.query_size = query_size
        self.train_size = train_size
        
        self.train_transform = get_transform("train")
        self.query_transform = get_transform("query")
        self.database_transform = get_transform("database")
        
        self.cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
        self.cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
        self.test_indices = list(range(len(self.cifar10_test)))
        self.train_indices = list(range(len(self.cifar10_train)))
        self.query_dataset, self.train_dataset, self.database_dataset = self.create_dataset()
    
    def create_dataset(self):
        query_indices = random.sample(self.test_indices, self.query_size)
        train_indices = random.sample(self.train_indices, self.train_size)
        
        remaining_test_indices = [i for i in self.test_indices if i not in query_indices]
        remaining_train_indices = [i for i in self.train_indices if i not in train_indices]
        
        query_dataset = TransformSubset(self.cifar10_test, query_indices, transform=self.query_transform)
        
        database_test_dataset = TransformSubset(
            self.cifar10_train, remaining_test_indices, transform=self.database_transform
        )
        database_train_dataset = TransformSubset(
            self.cifar10_train, remaining_train_indices, transform=self.database_transform
        )
        database_dataset = torch.utils.data.ConcatDataset([
            database_train_dataset,
            database_test_dataset
        ])
        
        
        train_dataset = TransformSubset(
            self.cifar10_train, train_indices, transform=self.train_transform
        )
        
        return query_dataset, train_dataset, database_dataset
    
    def __len__(self):
        if self.mode == "train":
            return len(self.train_dataset)
        if self.mode == "query":
            return len(self.query_dataset)
        return len(self.database_dataset)
    
    def __getitem__(self, index):
        if self.mode == "train":
            return self.train_dataset[index]
        if self.mode == "query":
            return self.query_dataset[index]
        return self.database_dataset[index]