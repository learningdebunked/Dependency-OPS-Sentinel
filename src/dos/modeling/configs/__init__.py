""
Model and training configurations (Experimental)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml

@dataclass
class ModelConfig:
    """Base configuration for model architecture."""
    model_type: str = "lstm_autoencoder"
    input_dim: int = 10
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    dropout: float = 0.1
    activation: str = "relu"
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ModelConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.__dict__

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data loading
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Logging
    log_interval: int = 10
    save_interval: int = 5
    
    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.__dict__

# Example configuration files are in configs/example_config.yaml
