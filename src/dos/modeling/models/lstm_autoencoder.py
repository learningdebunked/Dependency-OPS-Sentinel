"""
LSTM-based Autoencoder for anomaly detection (Experimental)
"""
import torch
import torch.nn as nn
from typing import Dict, Any
from ...ingestion.schemas import SignalType
from ..base_model import BaseModel

class LSTMAutoencoder(BaseModel):
    """
    LSTM-based Autoencoder for time series anomaly detection.
    
    Args:
        config: Model configuration dictionary
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.input_dim = config.get('input_dim', 10)
        self.hidden_dims = config.get('hidden_dims', [64, 32, 16])
        self.latent_dim = config.get('latent_dim', 8)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.1)
        
        super().__init__(config)
        
    def _setup_model(self) -> None:
        """Initialize model architecture."""
        # Encoder
        self.encoder = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dims[0],
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # Latent space
        self.latent = nn.Linear(self.hidden_dims[0], self.latent_dim)
        
        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=self.latent_dim,
            hidden_size=self.hidden_dims[0],
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        self.decoder_fc = nn.Linear(self.hidden_dims[0], self.input_dim)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input sequence to latent space."""
        _, (h_n, _) = self.encoder(x)
        # Use the last layer's hidden state
        latent = self.latent(h_n[-1])
        return latent
    
    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode latent vector to reconstructed sequence."""
        # Expand latent vector for sequence generation
        z = z.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Pass through decoder LSTM
        output, _ = self.decoder_lstm(z)
        
        # Final projection
        reconstructed = self.decoder_fc(output)
        return reconstructed
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the autoencoder."""
        batch_size, seq_len, _ = x.size()
        
        # Encode to latent space
        z = self.encode(x)
        
        # Decode back to sequence
        x_recon = self.decode(z, seq_len)
        
        return {
            'reconstructed': x_recon,
            'latent': z,
            'anomaly_score': self.compute_anomaly_score(x, x_recon)
        }
    
    def compute_anomaly_score(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score based on reconstruction error."""
        # Mean squared error across features
        mse = torch.mean((x - x_recon) ** 2, dim=-1)
        # Max over time steps
        return torch.max(mse, dim=1)[0]
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'LSTMAutoencoder':
        """Create model from configuration."""
        return cls(config)
