"""
Hello World example for Dependency Ops Sentinel (DOS)

This minimal example shows how to set up basic monitoring and alerting with DOS.
"""
import time
import random
import yaml
from datetime import datetime

class DOSHelloWorld:
    def __init__(self, config_path='config.yaml'):
        """Initialize with configuration."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'latency_ms': 0.0,
            'requests_processed': 0,
            'errors': 0
        }
        
        print("🚀 DOS Hello World initialized!")
        print(f"Monitoring service: {list(self.config['services'].keys())[0]}")

    def generate_metrics(self):
        """Generate synthetic metrics."""
        # Simulate normal operation with occasional spikes
        base_cpu = random.uniform(10, 30)
        spike = 70 if random.random() < 0.1 else 0  # 10% chance of spike
        
        self.metrics.update({
            'cpu_usage': min(100, base_cpu + spike),
            'memory_usage': random.uniform(200, 400),  # MB
            'latency_ms': random.uniform(50, 150),
            'requests_processed': self.metrics['requests_processed'] + 1,
            'errors': self.metrics['errors'] + (1 if random.random() < 0.01 else 0)  # 1% error rate
        })
        return self.metrics

    def check_alerts(self, metrics):
        """Check if any alerts should be triggered."""
        alerts = []
        
        if metrics['cpu_usage'] > self.config['alerting']['cpu_threshold']:
            alerts.append(f"High CPU usage: {metrics['cpu_usage']:.1f}%")
            
        if metrics['memory_usage'] > self.config['alerting']['memory_threshold']:
            alerts.append(f"High memory usage: {metrics['memory_usage']:.1f}%")
            
        if metrics['latency_ms'] > self.config['alerting']['latency_threshold']:
            alerts.append(f"High latency: {metrics['latency_ms']:.1f}ms")
            
        return alerts

    def run(self, duration=60):
        """Run the example for the specified duration (seconds)."""
        print(f"\n📊 Starting monitoring for {duration} seconds...\n")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            # Generate and check metrics
            metrics = self.generate_metrics()
            alerts = self.check_alerts(metrics)
            
            # Print current status
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] CPU: {metrics['cpu_usage']:5.1f}% | "
                  f"MEM: {metrics['memory_usage']:5.1f}MB | "
                  f"LAT: {metrics['latency_ms']:5.1f}ms | "
                  f"REQ: {metrics['requests_processed']}")
            
            # Print alerts if any
            for alert in alerts:
                print(f"   🚨 ALERT: {alert}")
            
            # Wait for next interval
            time.sleep(1)
        
        print("\n✅ Monitoring complete!")
        print(f"Total requests: {self.metrics['requests_processed']}")
        print(f"Total errors: {self.metrics['errors']} ({self.metrics['errors']/self.metrics['requests_processed']*100:.1f}%)")

if __name__ == "__main__":
    # Run the example
    dos = DOSHelloWorld()
    dos.run(duration=30)  # Run for 30 seconds
