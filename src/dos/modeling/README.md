# DOS Modeling Module (Experimental)

This module provides tools for training, evaluating, and deploying machine learning models for anomaly detection and root cause analysis in distributed systems.

⚠️ **Status**: Experimental - APIs may change in future releases

## Features

- 🏗️ **Flexible Model Architecture**: Support for various model types (LSTM, Transformer, etc.)
- 📊 **Data Pipeline**: Tools for loading, preprocessing, and batching time series data
- 🚀 **Training Utilities**: Distributed training, mixed precision, and early stopping
- 📈 **Experiment Tracking**: Integration with TensorBoard for visualization
- 🔄 **Model Serving**: Easy export to production formats

## Directory Structure

```
modeling/
├── __init__.py           # Module exports
├── base_model.py         # Abstract base model class
├── configs/              # Configuration files
│   ├── __init__.py
│   └── example_config.yaml
├── data/                 # Data loading utilities
│   └── dataset.py
├── models/               # Model implementations
│   ├── __init__.py
│   └── lstm_autoencoder.py
├── trainer.py            # Training loop and utilities
└── utils/                # Helper functions
    └── __init__.py
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Optional: CUDA-enabled GPU for faster training

### Installation

1. Install the required dependencies:

```bash
pip install torch numpy pandas pyyaml tqdm tensorboard
```

### Example Usage

1. **Define a model configuration** (`config.yaml`):

```yaml
model:
  model_type: "lstm_autoencoder"
  input_dim: 10
  hidden_dims: [64, 32, 16]
  latent_dim: 8
  num_layers: 2
  dropout: 0.1

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 10
```

2. **Create and train a model**:

```python
from dos.modeling import (
    LSTMAutoencoder,
    ModelConfig,
    TrainingConfig,
    DOSDataset,
    train_model
)
import numpy as np

# Generate sample data
num_samples = 1000
seq_len = 32
num_features = 10
data = np.random.randn(num_samples, seq_len, num_features)

# Create dataset
dataset = DOSDataset(data, window_size=seq_len)

# Initialize model and trainer
model = LSTMAutoencoder({
    'input_dim': num_features,
    'hidden_dims': [64, 32, 16],
    'latent_dim': 8,
    'num_layers': 2,
    'dropout': 0.1
})

# Train the model
trained_model, history = train_model(
    model=model,
    dataset=dataset,
    epochs=100,
    batch_size=32,
    learning_rate=1e-3
)
```

## Model Zoo

### Available Models

1. **LSTMAutoencoder**
   - LSTM-based autoencoder for time series anomaly detection
   - Handles variable-length sequences
   - Configurable architecture (hidden layers, dropout, etc.)

## Advanced Usage

### Custom Models

To create a custom model, inherit from `BaseModel` and implement the required methods:

```python
from dos.modeling.base_model import BaseModel
import torch.nn as nn

class MyCustomModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your model layers here
        self.layer1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.layer2 = nn.Linear(config.hidden_dim, config.output_dim)
    
    def forward(self, x):
        # Define forward pass
        x = torch.relu(self.layer1(x))
        return self.layer2(x)
```

### Custom Training Loop

For more control, you can write a custom training loop:

```python
trainer = ModelTrainer(model, config)

for epoch in range(config.epochs):
    # Training
    model.train()
    for batch in train_loader:
        loss = trainer.train_step(batch)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = trainer.validate(val_loader)
    
    # Log metrics
    trainer.log_metrics(epoch, {'train_loss': loss, 'val_loss': val_loss})
    
    # Save checkpoint
    if trainer.should_save_checkpoint():
        trainer.save_checkpoint()
```

## Contributing

Contributions are welcome! Please see the main project's contributing guidelines for details.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
