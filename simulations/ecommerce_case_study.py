"""
E-commerce Platform Case Study
=============================

This module contains a realistic simulation of an e-commerce platform's
microservices architecture, using patterns observed in production systems.
"""

import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np

class EcommerceDataset:
    """Generates realistic e-commerce platform metrics and logs."""
    
    def __init__(self):
        self.services = [
            "api-gateway",
            "user-service",
            "product-service",
            "cart-service",
            "order-service",
            "payment-service",
            "inventory-service",
            "notification-service",
            "search-service",
            "analytics-service"
        ]
        self.users = [f"user_{i}@example.com" for i in range(1, 1001)]
        self.products = [f"prod_{i:04d}" for i in range(1, 501)]
        self.regions = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"]
        self.environments = ["production", "staging", "canary"]
        self.versions = ["v1.0.0", "v1.1.0", "v2.0.0"]
        
    def generate_metrics(self, start_time: datetime, duration_hours: int = 24) -> List[Dict]:
        """Generate a day's worth of metrics with realistic patterns."""
        metrics = []
        current = start_time
        
        for _ in range(duration_hours * 12):  # 5-minute intervals
            timestamp = current.isoformat()
            hour = current.hour
            weekday = current.weekday()
            
            # Base traffic pattern (higher during business hours)
            time_factor = 0.3 + 0.7 * (np.sin((hour - 9) * np.pi/16) ** 2)
            day_factor = 1.5 if weekday < 5 else 0.7  # Weekdays vs weekend
            base_rps = random.uniform(800, 1200) * time_factor * day_factor
            
            for service in self.services:
                # Service-specific adjustments
                if service == "api-gateway":
                    rps = base_rps * random.uniform(0.8, 1.2)
                    error_rate = random.uniform(0.001, 0.005)
                    latency = random.uniform(50, 150)
                elif service == "payment-service":
                    rps = base_rps * 0.2 * random.uniform(0.7, 1.3)
                    error_rate = random.uniform(0.0005, 0.002)
                    latency = random.uniform(100, 300)
                else:
                    rps = base_rps * random.uniform(0.3, 0.8)
                    error_rate = random.uniform(0.001, 0.01)
                    latency = random.uniform(20, 100)
                
                metrics.append({
                    "timestamp": timestamp,
                    "service": service,
                    "rps": max(0, rps + random.normalvariate(0, rps*0.1)),
                    "error_rate": max(0, min(1, error_rate + random.normalvariate(0, error_rate*0.5))),
                    "latency_ms": max(10, latency + random.normalvariate(0, latency*0.2)),
                    "cpu_usage": random.uniform(0.1, 0.6),
                    "memory_usage": random.uniform(0.2, 0.7),
                    "region": random.choice(self.regions),
                    "version": random.choice(self.versions),
                    "environment": random.choices(
                        self.environments,
                        weights=[0.85, 0.1, 0.05]  # 85% prod, 10% staging, 5% canary
                    )[0]
                })
            
            current += timedelta(minutes=5)
        
        return metrics
    
    def generate_incidents(self, start_time: datetime, duration_days: int = 7) -> List[Dict]:
        """Generate realistic incident data."""
        incidents = []
        current = start_time
        
        # Known patterns of incidents
        patterns = [
            {"type": "deployment", "services": ["api-gateway", "user-service"], "duration": 30},
            {"type": "database", "services": ["product-service", "inventory-service"], "duration": 90},
            {"type": "network", "services": ["payment-service", "order-service"], "duration": 15},
            {"type": "memory_leak", "services": ["analytics-service"], "duration": 240},
            {"type": "config_error", "services": ["cart-service"], "duration": 45}
        ]
        
        for day in range(duration_days):
            # Business hours incident (more likely)
            if random.random() < 0.3:  # 30% chance per day
                incident_time = current.replace(hour=random.randint(10, 16), minute=random.randint(0, 59))
                pattern = random.choice(patterns)
                
                incident = {
                    "start_time": incident_time.isoformat(),
                    "end_time": (incident_time + timedelta(minutes=pattern["duration"])).isoformat(),
                    "type": pattern["type"],
                    "services": pattern["services"],
                    "severity": random.choice(["low", "medium", "high"]),
                    "description": f"{pattern['type'].replace('_', ' ').title()} issue affecting {', '.join(pattern['services'])}",
                    "root_cause": random.choice([
                        "Configuration drift",
                        "Network partition",
                        "Resource exhaustion",
                        "Software bug",
                        "Dependency failure"
                    ]),
                    "detected_by": random.choice(["alerting", "user_report", "monitoring"]),
                    "time_to_detect": random.randint(30, 300),  # seconds
                    "time_to_resolve": pattern["duration"] * 60  # seconds
                }
                incidents.append(incident)
            
            current += timedelta(days=1)
        
        return incidents
    
    def generate_deployment_logs(self, start_time: datetime, days: int = 7) -> List[Dict]:
        """Generate deployment and configuration change logs."""
        logs = []
        current = start_time
        
        for day in range(days):
            # 1-3 deployments per day
            for _ in range(random.randint(1, 3)):
                service = random.choice(self.services)
                env = random.choices(
                    ["production", "staging", "canary"],
                    weights=[0.6, 0.3, 0.1]
                )[0]
                
                log = {
                    "timestamp": current.isoformat(),
                    "service": service,
                    "environment": env,
                    "version": random.choice(self.versions),
                    "type": "deployment",
                    "status": random.choices(
                        ["success", "failed", "rolled_back"],
                        weights=[0.9, 0.08, 0.02]
                    )[0],
                    "duration_seconds": random.randint(30, 300),
                    "initiated_by": f"user_{random.randint(1, 50)}@example.com",
                    "commit_id": f"{random.getrandbits(64):016x}",
                    "config_changes": [
                        {
                            "key": random.choice(["db.pool.size", "cache.ttl", "timeout", "batch.size"]),
                            "old_value": str(random.randint(1, 100)),
                            "new_value": str(random.randint(1, 100))
                        }
                        for _ in range(random.randint(1, 3))
                    ]
                }
                logs.append(log)
                current += timedelta(minutes=random.randint(5, 120))
            
            current = (current + timedelta(days=1)).replace(hour=0, minute=0, second=0)
        
        return logs

def generate_case_study(output_dir: str = "case_study_data"):
    """Generate a complete case study dataset."""
    import os
    import json
    from datetime import datetime, timezone
    
    os.makedirs(output_dir, exist_ok=True)
    dataset = EcommerceDataset()
    
    # Generate one week of data
    start_time = datetime(2023, 6, 1, tzinfo=timezone.utc)
    
    # 1. Metrics (5-min intervals for 7 days)
    metrics = dataset.generate_metrics(start_time, duration_hours=24*7)
    with open(f"{output_dir}/metrics.jsonl", "w") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")
    
    # 2. Incidents
    incidents = dataset.generate_incidents(start_time, duration_days=7)
    with open(f"{output_dir}/incidents.json", "w") as f:
        json.dump(incidents, f, indent=2)
    
    # 3. Deployment logs
    deployments = dataset.generate_deployment_logs(start_time, days=7)
    with open(f"{output_dir}/deployments.json", "w") as f:
        json.dump(deployments, f, indent=2)
    
    print(f"Generated case study data in {output_dir}/")
    print(f"- {len(metrics)} metric samples")
    print(f"- {len(incidents)} incidents")
    print(f"- {len(deployments)} deployments")

if __name__ == "__main__":
    generate_case_study()
