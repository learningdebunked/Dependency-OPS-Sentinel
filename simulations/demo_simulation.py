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

class ConfigChangeGenerator:
    """Generate configuration changes and feature flags."""
    
    def __init__(self):
        self.config_versions = {}
        self.feature_flags = {}
        self.change_history = []
        
    def generate_config_change(self, service: str) -> Dict:
        """Generate a random configuration change."""
        change_types = [
            "database_connection_pool",
            "timeout_settings",
            "cache_config",
            "rate_limiting",
            "security_policies"
        ]
        change = {
            "type": "config_update",
            "service": service,
            "config_key": random.choice(change_types),
            "old_value": random.randint(1, 100),
            "new_value": random.randint(1, 100),
            "timestamp": datetime.utcnow().isoformat(),
            "source": "git" if random.random() > 0.5 else "kubernetes"
        }
        self.change_history.append(change)
        return change
        
    def toggle_feature_flag(self, service: str) -> Dict:
        """Toggle a feature flag for a service."""
        flag_name = f"feature_{random.choice(['dark_mode', 'new_checkout', 'ab_test'])}"
        current = self.feature_flags.get((service, flag_name), False)
        self.feature_flags[(service, flag_name)] = not current
        return {
            "type": "feature_flag",
            "service": service,
            "flag": flag_name,
            "enabled": not current,
            "timestamp": datetime.utcnow().isoformat()
        }

class CloudEventGenerator:
    """Generate cloud provider events."""
    
    def __init__(self):
        self.regions = ["us-west-1", "us-east-1", "eu-west-1", "ap-southeast-1"]
        self.event_types = [
            "instance_termination",
            "scheduled_maintenance",
            "network_connectivity",
            "capacity_change",
            "credential_rotation"
        ]
        
    def generate_cloud_event(self) -> Dict:
        """Generate a random cloud provider event."""
        event_type = random.choice(self.event_types)
        return {
            "type": "cloud_event",
            "event_type": event_type,
            "region": random.choice(self.regions),
            "severity": random.choice(["info", "warning", "critical"]),
            "timestamp": datetime.utcnow().isoformat(),
            "description": f"Cloud event: {event_type} in {random.choice(self.regions)}"
        }

class IncidentSimulator:
    """Simulate different types of incidents."""
    
    def __init__(self):
        self.incident_types = [
            self._simulate_cpu_spike,
            self._simulate_memory_leak,
            self._simulate_network_latency,
            self._simulate_error_rate_increase,
            self._simulate_cascading_failure,
            self._simulate_deployment_issue,
            self._simulate_config_mismatch
        ]
        self.current_incident = None
        self.incident_end_time = None
        self.config_generator = ConfigChangeGenerator()
        self.cloud_generator = CloudEventGenerator()
    
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
        
    def _simulate_deployment_issue(self, service: str, duration: int) -> Dict:
        return {
            "type": "deployment_issue",
            "service": service,
            "severity": "high",
            "description": f"Deployment issue in {service}: {random.choice(['ImagePullBackOff', 'CrashLoopBackOff', 'ImageNotFound'])}",
            "start_time": datetime.utcnow().isoformat(),
            "duration_seconds": duration
        }
        
    def _simulate_config_mismatch(self, service: str, duration: int) -> Dict:
        return {
            "type": "config_mismatch",
            "service": service,
            "severity": "medium",
            "description": f"Configuration mismatch in {service}: {random.choice(['Environment variables', 'Secrets', 'Resource limits'])} not synchronized",
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
        self.config_change_probability = 0.05  # 5% chance of config change
        self.feature_flag_probability = 0.03  # 3% chance of feature flag change
        self.cloud_event_probability = 0.02  # 2% chance of cloud event
        self.current_incident = None
        self.last_config_change = {}
        self.active_feature_flags = {}
        self.cloud_events = []
    
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
    
    def _print_event(self, event_type: str, message: str, color: str = "white"):
        """Print an event message with color coding."""
        colors = {
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "reset": "\033[0m"
        }
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        print(f"{colors.get(color, '')}[{timestamp}] {event_type.upper()}: {message}{colors['reset']}")

    async def _simulation_cycle(self):
        """Run one cycle of the simulation."""
        timestamp = datetime.utcnow()
        
        # Generate metrics for all services
        metrics = {}
        for service in self.services:
            metrics[service] = self.service_generators[service].generate_metrics(timestamp)
            
            # Apply any active incident effects
            if self.current_incident and self.current_incident["service"] == service:
                metrics[service] = self.incident_simulator.update_metrics(metrics[service])
            
            # Simulate config changes
            if random.random() < self.config_change_probability:
                change = self.incident_simulator.config_generator.generate_config_change(service)
                self.last_config_change[service] = change
                self._print_event("config", 
                               f"{service}: {change['config_key']} changed from {change['old_value']} to {change['new_value']}", 
                               "yellow")
                    
            # Simulate feature flag changes
            if random.random() < self.feature_flag_probability:
                flag_change = self.incident_simulator.config_generator.toggle_feature_flag(service)
                self.active_feature_flags[(service, flag_change['flag'])] = flag_change['enabled']
                status = "enabled" if flag_change['enabled'] else "disabled"
                self._print_event("feature", 
                               f"{service}: {flag_change['flag']} {status}", 
                               "blue")
        
        # Simulate cloud events
        if random.random() < self.cloud_event_probability:
            cloud_event = self.incident_simulator.cloud_generator.generate_cloud_event()
            self.cloud_events.append(cloud_event)
            self._print_event("cloud", 
                           f"{cloud_event['event_type']} in {cloud_event['region']} ({cloud_event['severity']})", 
                           "red" if cloud_event['severity'] == 'critical' else "yellow")
        
        # Print service status
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

class TestScenarios:
    """Predefined test scenarios for the simulation."""
    
    @staticmethod
    async def rolling_update_scenario(simulation):
        """Simulate a rolling update with potential issues."""
        print("\n🚀 Starting Rolling Update Scenario")
        print("1. Starting canary deployment of user-service v2.0.0")
        
        # Simulate canary deployment
        simulation.config_generator.generate_config_change("user-service")
        await asyncio.sleep(5)
        
        # Introduce a configuration issue
        print("2. Introducing configuration mismatch...")
        simulation.incident_simulator.trigger_incident(["user-service"])
        simulation.incident_simulator.current_incident["type"] = "config_mismatch"
        await asyncio.sleep(10)
        
        # Simulate rollback
        print("3. Rolling back due to issues...")
        simulation.config_generator.generate_config_change("user-service")
        simulation.incident_simulator.current_incident = None
        print("✅ Rolling Update Scenario Complete\n")
    
    @staticmethod
    async def feature_rollout_scenario(simulation):
        """Simulate a feature flag rollout with monitoring."""
        print("\n🚀 Starting Feature Rollout Scenario")
        print("1. Enabling new checkout flow for 10% of users")
        
        # Enable feature flag for a percentage of users
        simulation.config_generator.toggle_feature_flag("checkout-service")
        await asyncio.sleep(5)
        
        # Simulate increased load
        print("2. Simulating increased traffic...")
        simulation.incident_simulator.trigger_incident(["checkout-service"])
        simulation.incident_simulator.current_incident["type"] = "high_traffic"
        await asyncio.sleep(8)
        
        # Auto-scale and complete rollout
        print("3. Auto-scaling and completing rollout")
        simulation.incident_simulator.current_incident = None
        print("✅ Feature Rollout Scenario Complete\n")
    
    @staticmethod
    async def cloud_outage_scenario(simulation):
        """Simulate a cloud region outage and failover."""
        print("\n🚀 Starting Cloud Outage Scenario")
        print("1. Simulating us-east-1 region outage")
        
        # Trigger cloud event
        outage = {
            "type": "cloud_event",
            "event_type": "region_outage",
            "region": "us-east-1",
            "severity": "critical",
            "timestamp": datetime.utcnow().isoformat(),
            "description": "Network connectivity issues in us-east-1"
        }
        simulation.cloud_events.append(outage)
        
        # Simulate failover
        await asyncio.sleep(5)
        print("2. Initiating failover to us-west-2")
        
        # Simulate recovery
        await asyncio.sleep(10)
        print("3. us-east-1 region restored")
        print("✅ Cloud Outage Scenario Complete\n")

async def run_scenario(scenario_name: str):
    """Run a specific test scenario."""
    simulation = DOSSimulation()
    scenario = getattr(TestScenarios, f"{scenario_name}_scenario", None)
    
    if not scenario:
        print(f"Scenario '{scenario_name}' not found!")
        return
        
    print(f"\n{'='*50}")
    print(f"🚀 RUNNING SCENARIO: {scenario_name.replace('_', ' ').title()}")
    print(f"{'='*50}\n")
    
    await scenario(simulation)

def main():
    """Run the simulation."""
    parser = argparse.ArgumentParser(description='Run DOS Simulation')
    parser.add_argument('--scenario', type=str, help='Name of the scenario to run',
                      choices=['rolling_update', 'feature_rollout', 'cloud_outage', 'all'])
    args = parser.parse_args()
    
    if args.scenario == 'all':
        scenarios = ['rolling_update', 'feature_rollout', 'cloud_outage']
        for scenario in scenarios:
            asyncio.run(run_scenario(scenario))
    elif args.scenario:
        asyncio.run(run_scenario(args.scenario))
    else:
        # Default to normal simulation mode
        simulation = DOSSimulation()
        asyncio.run(simulation.run_simulation())

if __name__ == "__main__":
    import argparse
    asyncio.run(main())
