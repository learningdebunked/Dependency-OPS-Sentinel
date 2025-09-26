import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import Dict, List, Optional
import numpy as np

class DependencyGraphAnalyzer(nn.Module):
    """
    Graph-based dependency analyzer using Graph Attention Networks.
    Models relationships and dependencies between system components.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=num_heads, dropout=0.2)
        self.conv2 = GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False)
        self.anomaly_scorer = nn.Linear(hidden_dim, 1)
        self.root_cause_identifier = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, edge_index):
        # Graph attention layers
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        
        # Generate anomaly scores and root cause probabilities
        anomaly_scores = torch.sigmoid(self.anomaly_scorer(x))
        root_cause_probs = torch.sigmoid(self.root_cause_identifier(x))
        
        return anomaly_scores, root_cause_probs

class TemporalAnalyzer(nn.Module):
    """
    Temporal pattern analyzer using LSTM for time-series data.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=4)
        self.output_layer = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Apply self-attention
        attn_out, _ = self.attention(
            lstm_out.transpose(0, 1),  # (seq_len, batch_size, hidden_dim*2)
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1)
        )
        # Weighted sum of attention outputs
        scores = self.output_layer(attn_out.transpose(0, 1))  # (batch_size, seq_len, 1)
        return scores.squeeze(-1)

class ImpactAnalyzer:
    """
    Analyzes business impact of incidents and provides recommendations.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.severity_thresholds = self.config.get('severity_thresholds', {
            'critical': 0.9,
            'high': 0.7,
            'medium': 0.5,
            'low': 0.3
        })
    
    def analyze_impact(self, anomaly_scores: np.ndarray, 
                      root_cause_probs: np.ndarray,
                      metadata: Dict) -> Dict:
        """
        Analyze the business impact of detected anomalies.
        
        Args:
            anomaly_scores: Array of anomaly scores (0-1)
            root_cause_probs: Array of root cause probabilities (0-1)
            metadata: Additional context about the system state
            
        Returns:
            Dict containing impact analysis and recommendations
        """
        max_anomaly = float(anomaly_scores.max())
        max_root_cause = float(root_cause_probs.max())
        
        # Determine severity level
        if max_anomaly >= self.severity_thresholds['critical']:
            severity = 'critical'
        elif max_anomaly >= self.severity_thresholds['high']:
            severity = 'high'
        elif max_anomaly >= self.severity_thresholds['medium']:
            severity = 'medium'
        else:
            severity = 'low'
            
        # Generate recommendations
        recommendations = self._generate_recommendations(
            severity, max_anomaly, max_root_cause, metadata
        )
        
        return {
            'severity': severity,
            'anomaly_score': max_anomaly,
            'root_cause_confidence': max_root_cause,
            'recommendations': recommendations,
            'timestamp': metadata.get('timestamp')
        }
    
    def _generate_recommendations(self, severity: str, anomaly_score: float,
                                root_cause_confidence: float,
                                metadata: Dict) -> List[str]:
        """Generate recommendations based on the analysis."""
        recommendations = []
        
        if severity in ['critical', 'high']:
            recommendations.append(
                f"Immediate action required: {severity.upper()} severity issue detected"
            )
            if root_cause_confidence > 0.7:
                recommendations.append(
                    f"High confidence ({root_cause_confidence*100:.1f}%) in root cause identification"
                )
            
            # Add specific recommendations based on metadata
            if 'component' in metadata:
                recommendations.append(
                    f"Affected component: {metadata['component']}"
                )
                
                if metadata.get('can_rollback', False):
                    recommendations.append(
                        "Recommended action: Rollback recent changes to this component"
                    )
                
                if metadata.get('has_redundancy', False):
                    recommendations.append(
                        "Recommended action: Failover to standby instance"
                    )
        
        return recommendations
