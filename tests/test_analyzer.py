import pytest
import torch
import numpy as np
from dos.core.analyzer import DependencyGraphAnalyzer, ImpactAnalyzer, TemporalAnalyzer

def test_dependency_graph_analyzer_initialization():
    """Test that the dependency graph analyzer initializes correctly."""
    input_dim = 10
    analyzer = DependencyGraphAnalyzer(input_dim=input_dim)
    
    # Check that all expected layers are initialized
    assert hasattr(analyzer, 'conv1')
    assert hasattr(analyzer, 'conv2')
    assert hasattr(analyzer, 'anomaly_scorer')
    assert hasattr(analyzer, 'root_cause_identifier')
    
    # Test forward pass with dummy data
    x = torch.randn(5, input_dim)  # 5 nodes with input_dim features
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)  # Simple chain graph
    
    anomaly_scores, root_cause_probs = analyzer(x, edge_index)
    
    # Check output shapes
    assert anomaly_scores.shape == (5, 1)  # One score per node
    assert root_cause_probs.shape == (5, 1)  # One probability per node
    
    # Check that outputs are in valid range (0-1)
    assert torch.all(anomaly_scores >= 0) and torch.all(anomaly_scores <= 1)
    assert torch.all(root_cause_probs >= 0) and torch.all(root_cause_probs <= 1)

def test_temporal_analyzer():
    """Test the temporal analyzer with LSTM and attention."""
    seq_len = 10
    input_dim = 5
    batch_size = 3
    
    analyzer = TemporalAnalyzer(input_dim=input_dim)
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, input_dim)
    scores = analyzer(x)
    
    # Check output shape
    assert scores.shape == (batch_size, seq_len)
    
    # Test that the model can handle different sequence lengths
    x = torch.randn(batch_size, seq_len * 2, input_dim)
    scores = analyzer(x)
    assert scores.shape == (batch_size, seq_len * 2)

def test_impact_analyzer():
    """Test the impact analyzer's scoring and recommendations."""
    analyzer = ImpactAnalyzer()
    
    # Test with critical severity
    result = analyzer.analyze_impact(
        anomaly_scores=np.array([0.95]),
        root_cause_probs=np.array([0.9]),
        metadata={
            'component': 'payment-service',
            'can_rollback': True,
            'has_redundancy': True,
            'timestamp': '2023-01-01T00:00:00Z'
        }
    )
    
    assert result['severity'] == 'critical'
    assert result['anomaly_score'] == 0.95
    assert 'rollback' in ' '.join(result['recommendations']).lower()
    
    # Test with low severity
    result = analyzer.analyze_impact(
        anomaly_scores=np.array([0.2]),
        root_cause_probs=np.array([0.3]),
        metadata={
            'component': 'analytics-service',
            'timestamp': '2023-01-01T00:00:00Z'
        }
    )
    
    assert result['severity'] == 'low'
    assert len(result['recommendations']) == 0  # No recommendations for low severity

@pytest.mark.parametrize("anomaly_score,expected_severity", [
    (0.95, 'critical'),
    (0.8, 'high'),
    (0.6, 'medium'),
    (0.2, 'low')
])
def test_impact_analyzer_severity_thresholds(anomaly_score, expected_severity):
    """Test that severity thresholds are applied correctly."""
    analyzer = ImpactAnalyzer()
    result = analyzer.analyze_impact(
        anomaly_scores=np.array([anomaly_score]),
        root_cause_probs=np.array([0.5]),
        metadata={'timestamp': '2023-01-01T00:00:00Z'}
    )
    assert result['severity'] == expected_severity

def test_impact_analyzer_with_multiple_anomalies():
    """Test that the analyzer handles multiple anomaly scores correctly."""
    analyzer = ImpactAnalyzer()
    
    # Should pick the highest severity
    result = analyzer.analyze_impact(
        anomaly_scores=np.array([0.4, 0.9, 0.2]),
        root_cause_probs=np.array([0.3, 0.8, 0.1]),
        metadata={
            'component': 'multi-service',
            'timestamp': '2023-01-01T00:00:00Z'
        }
    )
    
    assert result['severity'] == 'critical'  # Because of the 0.9 score
    assert result['anomaly_score'] == 0.9  # Should be the max score
