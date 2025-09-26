import pytest
import asyncio
from unittest.mock import MagicMock, patch
from dos.ingestion.data_ingestor import (
    DataIngestionManager,
    KubernetesIngestor,
    CloudMetricsIngestor
)

# Mock data for testing
MOCK_K8S_DATA = [
    {
        "pod_name": "test-pod-1",
        "namespace": "default",
        "cpu_usage": 0.6,
        "memory_usage": 150,
        "status": "Running",
        "restart_count": 0,
        "timestamp": "2023-01-01T00:00:00Z"
    }
]

MOCK_CLOUD_DATA = [
    {
        "instance_id": "i-12345678",
        "instance_type": "t3.medium",
        "region": "us-east-1",
        "metrics": {
            "CPUUtilization": 55.0,
            "NetworkIn": 1024,
            "NetworkOut": 2048
        },
        "status": "running",
        "timestamp": "2023-01-01T00:00:00Z"
    }
]

@pytest.mark.asyncio
async def test_kubernetes_ingestor():
    """Test the Kubernetes ingestor with mock data."""
    # Create a mock Kubernetes client
    with patch('dos.ingestion.data_ingestor.KubernetesIngestor.fetch_data') as mock_fetch:
        mock_fetch.return_value = MOCK_K8S_DATA
        
        ingestor = KubernetesIngestor({
            'namespace': 'test-namespace',
            'metrics': ['cpu_usage', 'memory_usage', 'pod_status']
        })
        
        # Test data collection
        data = await ingestor.process()
        
        # Verify the results
        assert len(data) == 1
        assert data[0]['source'] == 'kubernetes'
        assert data[0]['component_type'] == 'pod'
        assert data[0]['metrics']['cpu_usage'] == 0.6
        assert data[0]['metadata']['namespace'] == 'default'
        assert data[0]['metadata']['pod_name'] == 'test-pod-1'

@pytest.mark.asyncio
async def test_cloud_metrics_ingestor():
    """Test the Cloud metrics ingestor with mock data."""
    with patch('dos.ingestion.data_ingestor.CloudMetricsIngestor.fetch_data') as mock_fetch:
        mock_fetch.return_value = MOCK_CLOUD_DATA
        
        ingestor = CloudMetricsIngestor({
            'provider': 'aws',
            'regions': ['us-east-1'],
            'metrics': ['CPUUtilization', 'NetworkIn', 'NetworkOut']
        })
        
        # Test data collection
        data = await ingestor.process()
        
        # Verify the results
        assert len(data) == 1
        assert data[0]['source'] == 'aws_cloud'
        assert data[0]['component_type'] == 'cloud_instance'
        assert data[0]['metrics']['CPUUtilization'] == 55.0
        assert data[0]['metadata']['region'] == 'us-east-1'
        assert data[0]['metadata']['instance_id'] == 'i-12345678'

@pytest.mark.asyncio
async def test_data_ingestion_manager():
    """Test the data ingestion manager with multiple ingestors."""
    # Mock the process method of ingestors
    with patch('dos.ingestion.data_ingestor.KubernetesIngestor.process') as mock_k8s, \
         patch('dos.ingestion.data_ingestor.CloudMetricsIngestor.process') as mock_cloud:
        
        # Setup mock return values
        mock_k8s.return_value = [{'source': 'kubernetes', 'data': 'test'}]
        mock_cloud.return_value = [{'source': 'aws_cloud', 'data': 'test'}]
        
        # Initialize manager with test config
        manager = DataIngestionManager({
            'kubernetes': {'enabled': True, 'namespace': 'test'},
            'cloud_metrics': {'enabled': True, 'provider': 'aws', 'regions': ['us-east-1']}
        })
        
        # Test data collection
        results = await manager.collect_data()
        
        # Verify both ingestors were called
        assert 'kubernetes' in results
        assert 'aws_metrics' in results
        assert len(results['kubernetes']) > 0
        assert len(results['aws_metrics']) > 0

def test_kubernetes_ingestor_normalization():
    """Test the Kubernetes data normalization."""
    ingestor = KubernetesIngestor()
    normalized = ingestor.normalize_data(MOCK_K8S_DATA)
    
    assert len(normalized) == 1
    item = normalized[0]
    
    # Check structure
    assert set(item.keys()) == {'source', 'component_type', 'component_id', 'metrics', 'metadata'}
    
    # Check metrics
    metrics = item['metrics']
    assert set(metrics.keys()) == {'cpu_usage', 'memory_usage', 'status', 'restart_count'}
    
    # Check metadata
    metadata = item['metadata']
    assert metadata['pod_name'] == 'test-pod-1'
    assert metadata['namespace'] == 'default'

def test_cloud_metrics_ingestor_normalization():
    """Test the Cloud metrics data normalization."""
    ingestor = CloudMetricsIngestor({'provider': 'aws'})
    normalized = ingestor.normalize_data(MOCK_CLOUD_DATA)
    
    assert len(normalized) == 1
    item = normalized[0]
    
    # Check structure
    assert set(item.keys()) == {'source', 'component_type', 'component_id', 'metrics', 'metadata'}
    
    # Check metrics
    metrics = item['metrics']
    assert set(metrics.keys()) == {'CPUUtilization', 'NetworkIn', 'NetworkOut'}
    
    # Check metadata
    metadata = item['metadata']
    assert metadata['instance_id'] == 'i-12345678'
    assert metadata['region'] == 'us-east-1'
