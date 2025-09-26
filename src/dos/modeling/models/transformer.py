"""
Transformer-based model for time series anomaly detection (Experimental)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

from ...ingestion.schemas import SignalType
from ..base_model import BaseModel

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[:, :x.size(1)]

class TransformerModel(BaseModel):
    """
    Transformer-based model for time series anomaly detection.
    
    This model uses a transformer encoder to process time series data and
    reconstructs the input for anomaly detection based on reconstruction error.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the transformer model.
        
        Args:
            config: Configuration dictionary containing:
                - input_dim: Number of input features
                - d_model: Dimension of the model
                - nhead: Number of attention heads
                - num_layers: Number of transformer encoder layers
                - dim_feedforward: Dimension of feedforward network
                - dropout: Dropout probability
                - max_seq_len: Maximum sequence length
        """
        self.config = config
        super().__init__(config)
    
    def _setup_model(self) -> None:
        """Initialize the transformer architecture."""
        # Model parameters
        self.input_dim = self.config.get('input_dim', 10)
        self.d_model = self.config.get('d_model', 64)
        self.nhead = self.config.get('nhead', 4)
        self.num_layers = self.config.get('num_layers', 3)
        self.dim_feedforward = self.config.get('dim_feedforward', 256)
        self.dropout = self.config.get('dropout', 0.1)
        self.max_seq_len = self.config.get('max_seq_len', 100)
        
        # Input projection
        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, self.max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.d_model, self.input_dim)
        
        # Anomaly scoring head
        self.anomaly_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Dictionary containing:
                - reconstructed: Reconstructed input
                - anomaly_score: Anomaly score for each time step
                - attention: Attention weights (if return_attention=True)
        """
        # Input projection
        x_proj = self.input_proj(x)
        
        # Add positional encoding
        x_pe = self.pos_encoder(x_proj)
        
        # Transformer encoder
        encoder_output = self.transformer_encoder(x_pe)
        
        # Reconstruct input
        reconstructed = self.output_proj(encoder_output)
        
        # Calculate anomaly score
        anomaly_score = self.anomaly_head(encoder_output).squeeze(-1)
        
        return {
            'reconstructed': reconstructed,
            'anomaly_score': anomaly_score,
            'latent': encoder_output.mean(dim=1)  # Global average pooling
        }
    
    def compute_loss(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            x: Original input
            x_recon: Reconstructed input
            
        Returns:
            Loss value
        """
        # Mean squared error for reconstruction
        mse_loss = F.mse_loss(x_recon, x, reduction='none')
        
        # Sum over features, mean over sequence and batch
        return mse_loss.sum(dim=-1).mean()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'TransformerModel':
        """Create model from configuration."""
        return cls(config)
