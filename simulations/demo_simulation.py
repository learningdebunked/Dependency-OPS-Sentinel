"""
Dependency Ops Sentinel - Simulation Demo

This script simulates a microservices environment and demonstrates how DOS would detect
and respond to various types of incidents.
"""

import asyncio
import random
import time
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any, Optional
import json

# Mock data generators
class ServiceGenerator:
    """Generate realistic service metrics."""
    
    def __init__(self, service_name: str, baseline_cpu: float = 0.3, baseline_memory: float = 0.4):
        self.service_name = service_name
        self.baseline_cpu = baseline_cpu
        self.baseline_memory = baseline_memory
        self.request_rate = 100  # requests per second
        self.error_rate = 0.01  # 1% error rate
        self.latency_ms = 50  # average latency in ms
        
    def generate_metrics(self, timestamp: datetime) -> Dict[str, Any]:
        """Generate metrics for this service."""
        # Add some random variation
        cpu = max(0.1, min(0.9, self.baseline_cpu + random.uniform(-0.1, 0.1)))
        memory = max(0.2, min(0.8, self.baseline_memory + random.uniform(-0.05, 0.05)))
        
        # Simulate daily pattern
        hour = timestamp.hour
        daily_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (hour / 24))
        
        return {
            "timestamp": timestamp.isoformat(),
            "service": self.service_name,
            "cpu_usage": cpu * daily_factor,
            "memory_usage": memory,
            "request_rate": self.request_rate * daily_factor,
            "error_rate": self.error_rate,
            "latency_ms": self.latency_ms,
            "pod_count": 3,
            "healthy": True
        }

class IncidentSimulator:
    """Simulate different types of incidents."""
    
    def __init__(self):
        self.incident_types = [
            self._simulate_cpu_spike,
            self._simulate_memory_leak,
            self._simulate_network_latency,
            self._simulate_error_rate_increase,
            self._simulate_cascading_failure
        ]
        self.current_incident = None
        self.incident_end_time = None
    
    def trigger_incident(self, services: List[str]) -> Dict:
        """Trigger a random incident."""
        if self.current_incident:
            return self.current_incident
            
        incident_type = random.choice(self.incident_types)
        affected_service = random.choice(services)
        duration = random.randint(60, 300)  # 1-5 minutes
        
        self.current_incident = incident_type(affected_service, duration)
        self.incident_end_time = datetime.utcnow() + timedelta(seconds=duration)
        
        return self.current_incident
    
    def update_incident(self, metrics: Dict) -> Dict:
        """Update metrics based on current incident."""
        if not self.current_incident:
            return metrics
            
        if datetime.utcnow() >= self.incident_end_time:
            # Incident is over
            self.current_incident = None
            self.incident_end_time = None
            return metrics
            
        # Apply incident effects
        incident_type = self.current_incident["type"]
        affected_service = self.current_incident["service"]
        
        if metrics["service"] == affected_service:
            if incident_type == "cpu_spike":
                metrics["cpu_usage"] = min(0.95, metrics["cpu_usage"] * 3)
                metrics["latency_ms"] = metrics["latency_ms"] * 2
                
            elif incident_type == "memory_leak":
                metrics["memory_usage"] = min(0.9, metrics["memory_usage"] * 1.5)
                if metrics["memory_usage"] > 0.85:
                    metrics["pod_count"] = max(1, metrics["pod_count"] - 1)
                    
            elif incident_type == "network_latency":
                metrics["latency_ms"] = metrics["latency_ms"] * 5
                
            elif incident_type == "error_rate_increase":
                metrics["error_rate"] = min(0.5, metrics["error_rate"] * 10)
                
            elif incident_type == "cascading_failure":
                metrics["cpu_usage"] = min(0.98, metrics["cpu_usage"] * 2)
                metrics["error_rate"] = min(0.7, metrics["error_rate"] * 5)
                metrics["latency_ms"] = metrics["latency_ms"] * 3
                
                # Simulate cascading effect
                if random.random() < 0.3:  # 30% chance to affect dependent services
                    metrics["error_rate"] = min(0.3, metrics["error_rate"] * 2)
                    metrics["latency_ms"] = metrics["latency_ms"] * 1.5
        
        return metrics
    
    def _simulate_cpu_spike(self, service: str, duration: int) -> Dict:
        return {
            "type": "cpu_spike",
            "service": service,
            "severity": "high",
            "description": f"CPU spike detected in {service}",
            "start_time": datetime.utcnow().isoformat(),
            "duration_seconds": duration
        }
    
    def _simulate_memory_leak(self, service: str, duration: int) -> Dict:
        return {
            "type": "memory_leak",
            "service": service,
            "severity": "critical",
            "description": f"Memory leak detected in {service}",
            "start_time": datetime.utcnow().isoformat(),
            "duration_seconds": duration
        }
    
    def _simulate_network_latency(self, service: str, duration: int) -> Dict:
        return {
            "type": "network_latency",
            "service": service,
            "severity": "medium",
            "description": f"Network latency increased for {service}",
            "start_time": datetime.utcnow().isoformat(),
            "duration_seconds": duration
        }
    
    def _simulate_error_rate_increase(self, service: str, duration: int) -> Dict:
        return {
            "type": "error_rate_increase",
            "service": service,
            "severity": "high",
            "description": f"Error rate increased for {service}",
            "start_time": datetime.utcnow().isoformat(),
            "duration_seconds": duration
        }
    
    def _simulate_cascading_failure(self, service: str, duration: int) -> Dict:
        return {
            "type": "cascading_failure",
            "service": service,
            "severity": "critical",
            "description": f"Cascading failure starting from {service}",
            "start_time": datetime.utcnow().isoformat(),
            "duration_seconds": duration
        }

class DOSSimulation:
    """Main simulation class for DOS demonstration."""
    
    def __init__(self):
        self.services = [
            "api-gateway",
            "user-service",
            "order-service",
            "payment-service",
            "inventory-service",
            "notification-service",
            "analytics-service"
        ]
        self.service_generators = {name: ServiceGenerator(name) for name in self.services}
        self.incident_simulator = IncidentSimulator()
        self.incident_probability = 0.1  # 10% chance of an incident per cycle
        self.current_incident = None
    
    async def run_simulation(self, duration_minutes: int = 30):
        """Run the simulation for the specified duration."""
        print("🚀 Starting Dependency Ops Sentinel Simulation")
        print("=" * 60)
        print("Services being monitored:")
        for service in self.services:
            print(f"- {service}")
        print("\nPress Ctrl+C to stop the simulation\n")
        
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        try:
            while datetime.utcnow() < end_time:
                await self._simulation_cycle()
                await asyncio.sleep(2)  # 2-second cycle
                
        except KeyboardInterrupt:
            print("\nSimulation stopped by user")
        
        print("\n✅ Simulation completed")
    
    async def _simulation_cycle(self):
        """Run one cycle of the simulation."""
        current_time = datetime.utcnow()
        
        # Check if we should trigger a new incident
        if not self.incident_simulator.current_incident and random.random() < self.incident_probability:
            self.current_incident = self.incident_simulator.trigger_incident(self.services)
            self._print_incident_alert(self.current_incident)
        
        # Generate metrics for each service
        all_metrics = []
        for service_name, generator in self.service_generators.items():
            metrics = generator.generate_metrics(current_time)
            
            # Apply any active incidents
            metrics = self.incident_simulator.update_metrics(metrics)
            all_metrics.append(metrics)
            
            # Print status
            self._print_service_status(metrics)
        
        # Clear the screen for the next update
        print("\033[H\033[J", end="")  # ANSI escape codes to clear screen
        
        # Print current time and incident status
        print(f"🕒 {current_time.strftime('%Y-%m-%d %H:%M:%S')} | ", end="")
        if self.incident_simulator.current_incident:
            incident = self.incident_simulator.current_incident
            time_left = (self.incident_simulator.incident_end_time - current_time).total_seconds()
            print(f"🚨 ACTIVE INCIDENT: {incident['description']} (ends in {int(time_left)}s)")
        else:
            print("✅ All systems normal")
        
        print("-" * 60)
        print("SERVICE           | CPU   | MEM   | LATENCY | ERR RATE | PODS | STATUS")
        print("-" * 60)
    
    def _print_service_status(self, metrics: Dict):
        """Print the status of a service with color coding."""
        service = metrics["service"].ljust(15)
        cpu = f"{metrics['cpu_usage']*100:.1f}%".rjust(5)
        mem = f"{metrics['memory_usage']*100:.1f}%".rjust(5)
        latency = f"{metrics['latency_ms']:.0f}ms".rjust(7)
        error_rate = f"{metrics['error_rate']*100:.1f}%".rjust(7)
        pods = str(metrics['pod_count']).rjust(4)
        
        # Determine status and color
        status = "✅"
        if metrics['error_rate'] > 0.1:
            status = "🔴"
        elif metrics['cpu_usage'] > 0.8 or metrics['memory_usage'] > 0.8:
            status = "🟠"
        elif metrics['latency_ms'] > 200:
            status = "🟡"
        
        print(f"{service} | {cpu} | {mem} | {latency} | {error_rate} | {pods}  | {status}")
    
    def _print_incident_alert(self, incident: Dict):
        """Print an alert for a new incident."""
        print("\n" + "!" * 60)
        print(f"🚨 INCIDENT DETECTED: {incident['description']}")
        print(f"   Type: {incident['type']}")
        print(f"   Severity: {incident['severity'].upper()}")
        print(f"   Start Time: {incident['start_time']}")
        print(f"   Duration: {incident['duration_seconds']} seconds")
        print("!" * 60 + "\n")
        
        # Simulate DOS detection and response
        print("🔍 Dependency Ops Sentinel Analysis:")
        if incident["type"] == "cpu_spike":
            print("   - High CPU usage detected")
            print("   - Possible causes: Infinite loop, sudden traffic spike, or resource leak")
            print("   ✅ Recommended actions:")
            print("      * Scale up the service")
            print("      * Check for infinite loops")
            print("      * Review recent deployments")
            
        elif incident["type"] == "memory_leak":
            print("   - Memory leak detected")
            print("   - Possible causes: Unreleased resources, cache issues")
            print("   ✅ Recommended actions:")
            print("      * Restart affected pods")
            print("      * Check for memory leaks in recent code changes")
            print("      * Increase memory limits if needed")
            
        elif incident["type"] == "network_latency":
            print("   - Network latency increased")
            print("   - Possible causes: Network congestion, DNS issues, or service dependencies")
            print("   ✅ Recommended actions:")
            print("      * Check network connectivity between services")
            print("      * Verify DNS resolution")
            print("      * Review service dependencies")
            
        elif incident["type"] == "error_rate_increase":
            print("   - Error rate increased")
            print("   - Possible causes: Bug in recent deployment, dependency failure")
            print("   ✅ Recommended actions:")
            print("      * Rollback recent changes")
            print("      * Check logs for error patterns")
            print("      * Verify external dependencies")
            
        elif incident["type"] == "cascading_failure":
            print("   - Cascading failure detected")
            print("   - Possible causes: Service dependency chain failure, circuit breaker tripped")
            print("   🚨 IMMEDIATE ACTION REQUIRED:")
            print("      * Isolate failing service")
            print("      * Implement circuit breaking")
            print("      * Scale up dependent services")
            print("      * Rollback recent changes")
        
        print("\n📊 Mitigation in progress...\n")

# Add the update_metrics method to IncidentSimulator
setattr(IncidentSimulator, 'update_metrics', IncidentSimulator.update_incident)

async def main():
    """Run the simulation."""
    simulation = DOSSimulation()
    await simulation.run_simulation(duration_minutes=15)  # Run for 15 minutes

if __name__ == "__main__":
    asyncio.run(main())
