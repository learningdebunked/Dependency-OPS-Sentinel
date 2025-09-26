"""
Dependency Ops Sentinel - Modeling Module (Experimental)

This module provides tools for training, evaluating, and deploying machine learning models
for anomaly detection and root cause analysis in distributed systems.

⚠️ Status: Experimental - API may change in future releases
"""

__version__ = "0.1.0"
__status__ = "experimental"

# Import core components
from .base_model import BaseModel
from .configs import ModelConfig, TrainingConfig
from .data.dataset import DOSDataset
from .trainer import ModelTrainer

__all__ = [
    'BaseModel',
    'ModelConfig',
    'TrainingConfig',
    'DOSDataset',
    'ModelTrainer'
]
