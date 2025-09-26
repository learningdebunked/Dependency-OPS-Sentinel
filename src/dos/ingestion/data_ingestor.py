from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime
from abc import ABC, abstractmethod

class DataIngestor(ABC):
    """Abstract base class for data ingestion from various sources."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    async def fetch_data(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Fetch data from the source."""
        pass
    
    @abstractmethod
    def normalize_data(self, raw_data: Any) -> List[Dict[str, Any]]:
        """Normalize data into a standard format."""
        pass
    
    async def process(self) -> List[Dict[str, Any]]:
        """Fetch and normalize data."""
        try:
            raw_data = await self.fetch_data()
            normalized_data = self.normalize_data(raw_data)
            return normalized_data
        except Exception as e:
            self.logger.error(f"Error in data ingestion: {str(e)}", exc_info=True)
            raise

class KubernetesIngestor(DataIngestor):
    """Ingest data from Kubernetes API."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.namespace = self.config.get('namespace', 'default')
        self.metrics_to_collect = self.config.get('metrics', [
            'cpu_usage', 'memory_usage', 'pod_status', 'container_restarts'
        ])
    
    async def fetch_data(self, *args, **kwargs) -> List[Dict]:
        """Fetch data from Kubernetes API."""
        # In a real implementation, this would use the Kubernetes Python client
        # For now, we'll return mock data
        self.logger.info(f"Fetching Kubernetes metrics from namespace: {self.namespace}")
        
        # Mock data for demonstration
        return [
            {
                "pod_name": f"pod-{i}",
                "namespace": self.namespace,
                "cpu_usage": 0.5 + (i * 0.1),
                "memory_usage": 100 + (i * 50),
                "status": "Running",
                "restart_count": i % 3,
                "timestamp": datetime.utcnow().isoformat()
            }
            for i in range(1, 4)  # 3 sample pods
        ]
    
    def normalize_data(self, raw_data: List[Dict]) -> List[Dict[str, Any]]:
        """Normalize Kubernetes metrics data."""
        normalized = []
        for item in raw_data:
            normalized.append({
                "source": "kubernetes",
                "component_type": "pod",
                "component_id": f"{item['namespace']}/{item['pod_name']}",
                "metrics": {
                    "cpu_usage": item.get("cpu_usage", 0),
                    "memory_usage": item.get("memory_usage", 0),
                    "status": item.get("status", "Unknown"),
                    "restart_count": item.get("restart_count", 0)
                },
                "metadata": {
                    "namespace": item["namespace"],
                    "pod_name": item["pod_name"],
                    "timestamp": item.get("timestamp")
                }
            })
        return normalized

class CloudMetricsIngestor(DataIngestor):
    """Ingest data from cloud provider metrics (AWS CloudWatch, Azure Monitor, GCP Monitoring)."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.provider = self.config.get('provider', 'aws')
        self.regions = self.config.get('regions', ['us-east-1'])
        self.metrics = self.config.get('metrics', ['CPUUtilization', 'NetworkIn', 'NetworkOut'])
    
    async def fetch_data(self, *args, **kwargs) -> List[Dict]:
        """Fetch data from cloud provider's metrics service."""
        self.logger.info(f"Fetching {self.provider.upper()} metrics for regions: {', '.join(self.regions)}")
        
        # Mock data for demonstration
        return [
            {
                "instance_id": f"i-{i:08x}",
                "instance_type": "t3.medium",
                "region": region,
                "metrics": {
                    metric: 50 + (i * 5) + (j * 2)  # Some mock values
                    for j, metric in enumerate(self.metrics)
                },
                "status": "running",
                "timestamp": datetime.utcnow().isoformat()
            }
            for i, region in enumerate(self.regions)
            for _ in range(2)  # 2 instances per region
        ]
    
    def normalize_data(self, raw_data: List[Dict]) -> List[Dict[str, Any]]:
        """Normalize cloud metrics data."""
        normalized = []
        for item in raw_data:
            normalized.append({
                "source": f"{self.provider}_cloud",
                "component_type": "cloud_instance",
                "component_id": f"{self.provider}:{item['region']}:{item['instance_id']}",
                "metrics": item["metrics"],
                "metadata": {
                    "instance_id": item["instance_id"],
                    "instance_type": item["instance_type"],
                    "region": item["region"],
                    "status": item["status"],
                    "timestamp": item["timestamp"]
                }
            })
        return normalized

class DataIngestionManager:
    """Manages multiple data ingestion sources."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.ingestors = self._initialize_ingestors()
        self.logger = logging.getLogger(__name__)
    
    def _initialize_ingestors(self) -> Dict[str, DataIngestor]:
        """Initialize data ingestors based on configuration."""
        ingestors = {}
        
        # Kubernetes ingestor
        if self.config.get('kubernetes', {}).get('enabled', True):
            ingestors['kubernetes'] = KubernetesIngestor(
                self.config.get('kubernetes', {})
            )
        
        # Cloud metrics ingestor
        cloud_config = self.config.get('cloud_metrics', {})
        if cloud_config.get('enabled', True):
            provider = cloud_config.get('provider', 'aws')
            ingestors[f'{provider}_metrics'] = CloudMetricsIngestor(cloud_config)
        
        return ingestors
    
    async def collect_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Collect data from all configured ingestors."""
        results = {}
        
        for name, ingestor in self.ingestors.items():
            try:
                self.logger.info(f"Collecting data from {name}")
                results[name] = await ingestor.process()
                self.logger.info(f"Collected {len(results[name])} items from {name}")
            except Exception as e:
                self.logger.error(f"Failed to collect data from {name}: {str(e)}", exc_info=True)
                results[name] = []
        
        return results
