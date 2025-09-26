"""
Example training script for DOS modeling (Experimental)
"""
import os
import sys
import yaml
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.dos.modeling import (
    LSTMAutoencoder,
    ModelConfig,
    TrainingConfig,
    DOSDataset,
    train_model
)

def load_config(config_path: str) -> tuple:
    """Load model and training configurations."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = ModelConfig(**config['model'])
    training_config = TrainingConfig(**config['training'])
    
    return model_config, training_config

def generate_sample_data(
    num_samples: int = 1000,
    seq_len: int = 32,
    num_features: int = 10
) -> np.ndarray:
    """Generate sample time series data for demonstration."""
    # Simple sine waves with noise
    t = np.linspace(0, 10, num_samples * seq_len)
    data = np.zeros((num_samples, seq_len, num_features))
    
    for i in range(num_features):
        freq = 0.5 + 0.5 * i  # Different frequency for each feature
        signal = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
        
        # Reshape to (num_samples, seq_len)
        signal = signal.reshape(num_samples, seq_len, 1)
        data[..., i:i+1] = signal
    
    return data

def main():
    # Configuration
    config_path = "src/dos/modeling/configs/example_config.yaml"
    model_config, training_config = load_config(config_path)
    
    # Create sample dataset
    print("Generating sample data...")
    data = generate_sample_data(
        num_samples=1000,
        seq_len=32,
        num_features=model_config.input_dim
    )
    
    # Create dataset and data loaders
    dataset = DOSDataset(data, window_size=32, stride=1)
    
    # Initialize model
    print("Initializing model...")
    model = LSTMAutoencoder(model_config.__dict__)
    
    # Train the model
    print("Starting training...")
    trained_model, history = train_model(model, dataset, training_config)
    
    print("\nTraining complete!")
    print(f"Final training loss: {history['train_loss'][-1]:.6f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
    
    # Save the final model
    output_dir = Path(training_config.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trained_model.save(output_dir / "final_model.pt")
    print(f"Model saved to {output_dir / 'final_model.pt'}")

if __name__ == "__main__":
    main()
