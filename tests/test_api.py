import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json

from dos.api.app import app, data_manager

# Create a test client
client = TestClient(app)

# Test data
TEST_ANALYSIS_REQUEST = {
    "component_id": "test-service-1",
    "metrics": {
        "cpu_usage": 0.9,
        "memory_usage": 0.85,
        "latency_ms": 450,
        "error_rate": 0.15
    },
    "metadata": {
        "environment": "production",
        "can_rollback": True,
        "has_redundancy": False
    }
}

# Mock data for data collection
MOCK_COLLECTED_METRICS = {
    "kubernetes": [
        {
            "source": "kubernetes",
            "component_type": "pod",
            "component_id": "default/test-pod-1",
            "metrics": {
                "cpu_usage": 0.6,
                "memory_usage": 150,
                "status": "Running",
                "restart_count": 0
            },
            "metadata": {
                "namespace": "default",
                "pod_name": "test-pod-1",
                "timestamp": "2023-01-01T00:00:00Z"
            }
        }
    ],
    "aws_metrics": [
        {
            "source": "aws_cloud",
            "component_type": "cloud_instance",
            "component_id": "aws:us-east-1:i-12345678",
            "metrics": {
                "CPUUtilization": 55.0,
                "NetworkIn": 1024,
                "NetworkOut": 2048
            },
            "metadata": {
                "instance_id": "i-12345678",
                "instance_type": "t3.medium",
                "region": "us-east-1",
                "status": "running",
                "timestamp": "2023-01-01T00:00:00Z"
            }
        }
    ]
}

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data

@pytest.mark.asyncio
async def test_analyze_endpoint():
    """Test the analyze endpoint with a valid request."""
    with patch('dos.api.app.analyzer') as mock_analyzer, \
         patch('dos.api.app.impact_analyzer') as mock_impact_analyzer:
        
        # Setup mock responses
        mock_analyzer.return_value = (0.85, 0.75)  # anomaly_score, root_cause_confidence
        mock_impact_analyzer.analyze_impact.return_value = {
            'severity': 'critical',
            'recommendations': [
                'Immediate action required: CRITICAL severity issue detected',
                'Recommended action: Rollback recent changes to this component'
            ]
        }
        
        # Make the request
        response = client.post("/analyze", json=TEST_ANALYSIS_REQUEST)
        
        # Verify the response
        assert response.status_code == 200
        data = response.json()
        
        # Check the response structure
        assert "component_id" in data
        assert "anomaly_score" in data
        assert "root_cause_confidence" in data
        assert "severity" in data
        assert "recommendations" in data
        assert "timestamp" in data
        
        # Check specific values
        assert data["component_id"] == "test-service-1"
        assert data["severity"] == "critical"
        assert len(data["recommendations"]) > 0

@pytest.mark.asyncio
async def test_analyze_endpoint_invalid_request():
    """Test the analyze endpoint with an invalid request."""
    # Missing required field 'component_id'
    invalid_request = {"metrics": {"cpu_usage": 0.9}}
    response = client.post("/analyze", json=invalid_request)
    assert response.status_code == 422  # Validation error
    
    # Invalid metrics type
    invalid_request = {"component_id": "test", "metrics": "not_a_dict"}
    response = client.post("/analyze", json=invalid_request)
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_collect_metrics_endpoint():
    """Test the collect metrics endpoint."""
    # Mock the data manager's collect_data method
    with patch.object(data_manager, 'collect_data') as mock_collect:
        # Setup mock return value
        mock_collect.return_value = MOCK_COLLECTED_METRICS
        
        # Make the request
        response = client.get("/collect-metrics")
        
        # Verify the response
        assert response.status_code == 200
        data = response.json()
        
        # Check the response structure
        assert data["status"] == "success"
        assert "data" in data
        assert "kubernetes" in data["data"]
        assert "aws_metrics" in data["data"]
        
        # Check the data structure
        k8s_data = data["data"]["kubernetes"]
        assert len(k8s_data) == 1
        assert k8s_data[0]["component_type"] == "pod"
        
        aws_data = data["data"]["aws_metrics"]
        assert len(aws_data) == 1
        assert aws_data[0]["source"] == "aws_cloud"

@pytest.mark.asyncio
async def test_collect_metrics_error_handling():
    """Test error handling in the collect metrics endpoint."""
    # Mock the data manager's collect_data method to raise an exception
    with patch.object(data_manager, 'collect_data') as mock_collect:
        mock_collect.side_effect = Exception("Test error")
        
        # Make the request
        response = client.get("/collect-metrics")
        
        # Verify the error response
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Test error" in data["detail"]
