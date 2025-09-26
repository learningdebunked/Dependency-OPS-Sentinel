"""
Model training utilities (Experimental)
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from tqdm import tqdm

from .base_model import BaseModel
from .configs import TrainingConfig
from .data import create_data_loaders, DOSDataset

class ModelTrainer:
    """
    Handles model training, validation, and testing.
    """
    
    def __init__(self, model: BaseModel, config: TrainingConfig):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Create output directories
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.log_dir = Path(config.log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer and loss
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.criterion = nn.MSELoss()
        
        # Setup logging
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move data to device
            batch = batch.to(self.device)
            
            # Forward pass
            outputs = self.model(batch)
            loss = self.criterion(outputs['reconstructed'], batch)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: torch.utils.data.DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = batch.to(self.device)
                outputs = self.model(batch)
                loss = self.criterion(outputs['reconstructed'], batch)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader
    ) -> Dict[str, List[float]]:
        """
        Train the model with early stopping.
        
        Returns:
            Dictionary containing training history
        """
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(1, self.config.epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Log metrics
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            
            # Save checkpoint if validation loss improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint(f"best_model.pt")
            else:
                self.epochs_without_improvement += 1
            
            # Save periodic checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"Early stopping after {epoch} epochs")
                break
        
        return history
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'epoch': len(self.history['train_loss']) if hasattr(self, 'history') else 0,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.history['train_loss'][-1] if hasattr(self, 'history') else float('inf'),
            'val_loss': self.history['val_loss'][-1] if hasattr(self, 'history') else float('inf'),
            'config': self.model.config
        }, checkpoint_path)
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_class: type,
        config: TrainingConfig
    ) -> 'ModelTrainer':
        """Load trainer from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Initialize model with saved config
        model = model_class(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Initialize trainer
        trainer = cls(model, config)
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return trainer

def train_model(
    model: BaseModel,
    dataset: DOSDataset,
    config: TrainingConfig
) -> Tuple[BaseModel, Dict[str, List[float]]]:
    """
    Train a model with the given dataset and configuration.
    
    Args:
        model: Model to train
        dataset: Dataset for training/validation
        config: Training configuration
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(
        dataset,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        batch_size=config.batch_size
    )
    
    # Initialize trainer
    trainer = ModelTrainer(model, config)
    
    # Train the model
    history = trainer.train(train_loader, val_loader)
    
    return model, history
