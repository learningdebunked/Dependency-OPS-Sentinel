"""
Base model class for all DOS models (Experimental)
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all DOS models.
    
    All models should inherit from this class and implement the required methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model with configuration."""
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_model()
    
    @abstractmethod
    def _setup_model(self) -> None:
        """Initialize model architecture."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        pass
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """Load model from disk."""
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def to_device(self, device: str) -> None:
        """Move model to specified device."""
        self.device = torch.device(device)
        self.to(self.device)
