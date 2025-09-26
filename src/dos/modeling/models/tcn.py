"""
Temporal Convolutional Network (TCN) for time series anomaly detection (Experimental)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional

from ...ingestion.schemas import SignalType
from ..base_model import BaseModel

class TemporalBlock(nn.Module):
    """Temporal block for TCN."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """Initialize weights for better convergence."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the temporal block."""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    ""
    Chomp1d module for causal padding.
    
    This ensures that the output has the same length as the input
    by removing extra elements from the right side.
    """
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal padding."""
        return x[:, :, :-self.chomp_size].contiguous()

class TCNModel(BaseModel):
    """
    Temporal Convolutional Network for time series anomaly detection.
    
    This model uses dilated causal convolutions to capture temporal dependencies
    in time series data and detects anomalies based on reconstruction error.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the TCN model.
        
        Args:
            config: Configuration dictionary containing:
                - input_dim: Number of input features
                - num_channels: List of channel widths for each layer
                - kernel_size: Size of the convolutional kernel
                - dropout: Dropout probability
        """
        self.config = config
        super().__init__(config)
    
    def _setup_model(self) -> None:  # noqa: C901
        """Initialize the TCN architecture."""
        # Model parameters
        self.input_dim = self.config.get('input_dim', 10)
        self.num_channels = self.config.get('num_channels', [64, 64, 64, 64])
        self.kernel_size = self.config.get('kernel_size', 3)
        self.dropout = self.config.get('dropout', 0.2)
        
        # Create TCN layers
        layers = []
        num_levels = len(self.num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = self.input_dim if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]
            
            # Calculate padding for same-length output
            padding = (self.kernel_size - 1) * dilation
            
            layers += [
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    dropout=self.dropout
                )
            ]
        
        self.tcn = nn.Sequential(*layers)
        
        # Output projection
        self.output_proj = nn.Conv1d(
            self.num_channels[-1],
            self.input_dim,
            1
        )
        
        # Anomaly scoring head
        self.anomaly_head = nn.Sequential(
            nn.Conv1d(self.num_channels[-1], self.num_channels[-1] // 2, 1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Conv1d(self.num_channels[-1] // 2, 1, 1),
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
                - features: Extracted features
        """
        # Input shape: (batch_size, seq_len, input_dim)
        # Transpose to (batch_size, input_dim, seq_len) for Conv1d
        x = x.transpose(1, 2)
        
        # Pass through TCN
        features = self.tcn(x)
        
        # Reconstruct input
        reconstructed = self.output_proj(features).transpose(1, 2)
        
        # Calculate anomaly score
        anomaly_score = self.anomaly_head(features).squeeze(1)
        
        return {
            'reconstructed': reconstructed,
            'anomaly_score': anomaly_score,
            'features': features.transpose(1, 2)  # Back to (batch_size, seq_len, channels)
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
    def from_config(cls, config: Dict[str, Any]) -> 'TCNModel':
        """Create model from configuration."""
        return cls(config)
