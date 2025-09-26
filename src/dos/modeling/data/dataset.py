"""
Dataset classes for DOS modeling (Experimental)
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

class DOSDataset(Dataset):
    """
    Dataset class for DOS time series data.
    
    Args:
        data: Input data as numpy array of shape (n_samples, seq_len, n_features)
        window_size: Size of the sliding window
        stride: Stride for sliding window
    """
    
    def __init__(self, data: np.ndarray, window_size: int = 32, stride: int = 1):
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.length = (len(data) - window_size) // stride + 1
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.stride
        end = start + self.window_size
        window = self.data[start:end]
        return torch.FloatTensor(window)
    
    @classmethod
    def from_csv(cls, file_path: str, **kwargs) -> 'DOSDataset':
        """Load dataset from CSV file."""
        df = pd.read_csv(file_path)
        # Assuming first column is timestamp, rest are features
        data = df.iloc[:, 1:].values
        return cls(data, **kwargs)
    
    @classmethod
    def from_numpy(cls, file_path: str, **kwargs) -> 'DOSDataset':
        """Load dataset from numpy file."""
        data = np.load(file_path)
        return cls(data, **kwargs)

def create_data_loaders(
    dataset: DOSDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split dataset into train/val/test and create data loaders.
    
    Args:
        dataset: Input dataset
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
