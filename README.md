# Dependency Ops Sentinel (DOS)

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An intelligent dependency monitoring and incident response system that provides real-time anomaly detection, root cause analysis, and automated remediation for complex distributed systems.

## ✨ Features

- **Automated Dependency Monitoring**: Real-time monitoring of infrastructure and application dependencies
- **Anomaly Detection**: Machine learning-powered detection of abnormal behavior
- **Root Cause Analysis**: Identifies the source of issues in complex dependency graphs
- **Impact Assessment**: Evaluates the business impact of incidents
- **Automated Remediation**: Suggests and can execute remediation actions
- **Multi-Cloud Support**: Works across AWS, Azure, GCP, and on-premises environments
- **Kubernetes-Native**: Built with cloud-native technologies in mind
- **Real-time Alerts**: Configurable alerting for critical system events
- **Comprehensive Dashboard**: Visualize system health and dependencies

## 🏗️ Architecture

```mermaid
graph TD
    A[Data Sources] --> B[Data Ingestion Layer]
    B --> C[Analysis Engine]
    C --> D[Alerting System]
    C --> E[Dashboard]
    C --> F[Remediation Actions]
    
    subgraph Data Sources
    A1[Kubernetes]
    A2[Cloud Metrics]
    A3[Application Logs]
    A4[Custom Metrics]
    end
    
    subgraph Analysis Engine
    C1[Anomaly Detection]
    C2[Root Cause Analysis]
    C3[Impact Assessment]
    end
```

## 🎮 Simulation Environment

Dependency Ops Sentinel includes a realistic simulation environment that mimics production microservices behavior for testing and demonstration purposes.

### Key Features

#### 1. Realistic Service Behavior
- **Daily Traffic Patterns**: Simulates natural traffic fluctuations (day/night, week/weekend)
- **Resource Usage**: Realistic CPU/memory variations with configurable baselines
- **Service Dependencies**: Models complex inter-service relationships

#### 2. Incident Simulation
- **Multiple Incident Types**:
  - CPU Spikes
  - Memory Leaks
  - Network Latency
  - Error Rate Increases
  - Cascading Failures
- **Configurable Severity**: Control impact and duration of incidents

#### 3. Real-time Monitoring
- **Live Metrics**: View service health in real-time
- **Incident Alerts**: Get notified when issues are detected
- **Historical Data**: Analyze patterns and trends over time

### Running the Simulation

1. **Start the simulation**:
   ```bash
   python simulations/demo_simulation.py
   ```

2. **Observe the output**:
   ```
   [14:30:00] All services operating normally
   [14:31:15] 🚨 INCIDENT: CPU spike detected in payment-service (Duration: 180s)
   [14:32:00] 🔄 payment-service metrics: CPU=85% | Latency=200ms | Errors=2%
   [14:34:15] ✅ INCIDENT RESOLVED: payment-service CPU spike
   ```

3. **Customize the simulation**:
   - Modify `ServiceGenerator` for different baseline metrics
   - Adjust incident probabilities in `IncidentSimulator`
   - Add new service dependencies as needed

## 🔄 Detailed Simulation Flow

### 1. Initialization Phase
```mermaid
graph TD
    A[Main Application] -->|1. Initialize| B[Service Registry]
    A -->|2. Configure| C[Metric Generators]
    A -->|3. Setup| D[Incident Simulator]
    B -->|4. Register| E[Service Dependencies]
    
    subgraph Services
    B
    E
    end
    
    subgraph Configuration
    C
    D
    end
```
**Component Interactions**:
1. Main app loads service configurations
2. Creates ServiceGenerator instances for each service
3. Sets up IncidentSimulator with incident types
4. Builds dependency graph between services
5. Initializes metric baselines and thresholds

### 2. Metric Generation Loop
```mermaid
gantt
    title Metric Collection Cycle (1 Second)
    dateFormat  SS.000
    axisFormat %S.%L
    
    section Parallel Collection
    Service 1  :a1, 0.000, 0.2s
    Service 2  :a2, 0.000, 0.2s
    Service 3  :a3, 0.000, 0.2s
    
    section Aggregation
    Process Metrics :b1, after a1, 0.1s
    Apply Patterns  :b2, after b1, 0.1s
    
    section Output
    Publish :c1, after b2, 0.1s
```
**Data Flow**:
1. Each service generates its metrics in parallel
2. Applies daily/seasonal patterns
3. Adds random noise for realism
4. Aggregates metrics with timestamps

### 3. Incident Injection
```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> IncidentTriggered: Random(10%) chance
    
    state IncidentTriggered {
        [*] --> SelectType
        SelectType --> SelectTarget
        SelectTarget --> ApplyEffects
        ApplyEffects --> [*]
    }
    
    IncidentTriggered --> Idle: After duration
    
    note right of IncidentTriggered
        Types:
        - CPU Spike
        - Memory Leak
        - Network Latency
        - Error Rate
        - Cascading Failure
    end note
```
**Decision Process**:
1. Random roll for incident (10% chance)
2. Select incident type based on weights
3. Choose target service(s)
4. Calculate duration and severity
5. Apply initial effects

### 4. Impact Propagation
```mermaid
flowchart LR
    A[Incident Starts] --> B{Direct Impact}
    B -->|CPU/Memory| C[Service Metrics]
    B -->|Network| D[Communication]
    
    C --> E[Service Degradation]
    D --> F[Increased Latency]
    
    E --> G[Dependent Services]
    F --> G
    
    G --> H{Cascade Threshold?}
    H -->|Yes| I[Propagate]
    H -->|No| J[Contain]
    
    style B fill:#f9f,stroke:#333
    style H fill:#ff9,stroke:#333
```
**Propagation Rules**:
- Immediate effects on target service
- Network impacts affect communication
- Cascading failures follow dependency graph
- Thresholds prevent infinite propagation

### 5. Monitoring & Alerting
```mermaid
classDiagram
    class MetricMonitor {
        +metrics: dict
        +thresholds: dict
        +check_metrics()
        +trigger_alert()
    }
    
    class AlertManager {
        +alerts: list
        +deduplicate()
        +notify()
    }
    
    class NotificationChannel {
        +send_alert()
    }
    
    MetricMonitor --> AlertManager : reports
    AlertManager --> NotificationChannel : dispatches
    
    class EmailNotifier {
        +send_alert()
    }
    
    class SlackNotifier {
        +send_alert()
    }
    
    NotificationChannel <|-- EmailNotifier
    NotificationChannel <|-- SlackNotifier
```
**Alert Flow**:
1. Metrics compared against thresholds
2. Alerts generated for violations
3. Deduplication applied
4. Notifications dispatched via channels
5. Escalation if unacknowledged

### 6. Recovery & Cleanup
```mermaid
timeline
    title Incident Lifecycle
    section Incident Timeline
        Detection      : 0s
        Triage         : 5s
        Mitigation     : 10s
        Resolution     : 60s
        Post-Mortem    : 65s
    
    section System State
        Normal         : 0-5s
        Degraded       : 5-60s
        Recovering     : 60-65s
        Normal         : 65s+
    
    section Actions
        Alert          : 0s
        Auto-Remediate : 10s
        Verify        : 60s
        Cleanup       : 65s
```
**Recovery Process**:
1. Incident detected and logged
2. Automatic remediation attempts
3. Verification of resolution
4. Resource cleanup
5. Post-mortem analysis
6. Return to normal operations

### 2. Metric Generation Loop
```python
async def generate_metrics(self):
    while True:
        timestamp = datetime.utcnow()
        metrics = {}
        for service in self.services:
            metrics[service] = self.service_generators[service].generate_metrics(timestamp)
        await self.process_metrics(metrics)
        await asyncio.sleep(1)  # 1-second resolution
```
- **1-second Intervals**: Updates metrics every second
- **Parallel Collection**: Gathers metrics from all services simultaneously
- **Time Synchronization**: Ensures consistent timestamps across services

### 3. Incident Injection
```python
def trigger_incident(self):
    if random.random() < self.incident_probability:
        incident = self.incident_simulator.trigger_incident(self.services)
        self.current_incident = incident
        logger.warning(f"🚨 {incident['description']} (Duration: {incident['duration_seconds']}s)")
```
- **Random Triggering**: 10% chance per cycle to start an incident
- **Incident Selection**: Randomly selects from available incident types
- **Service Targeting**: Chooses a random service to be affected

### 4. Impact Propagation
```python
def apply_incident_effects(self, metrics):
    if not self.current_incident:
        return metrics
        
    affected_service = self.current_incident["service"]
    if affected_service in metrics:
        metrics[affected_service] = self.incident_simulator.update_metrics(
            metrics[affected_service]
        )
    return metrics
```
- **Direct Impact**: Applies primary effects to target service
- **Dependent Services**: May affect related services based on dependency graph
- **Metric Distortion**: Modifies metrics to reflect the incident

### 5. Monitoring & Alerting
```python
def check_thresholds(self, metrics):
    alerts = []
    for service, metric in metrics.items():
        if metric["cpu_usage"] > 0.8:
            alerts.append(f"High CPU in {service}: {metric['cpu_usage']:.0%}")
        if metric["error_rate"] > 0.1:
            alerts.append(f"High error rate in {service}: {metric['error_rate']:.1%}")
    return alerts
```
- **Threshold Checking**: Compares metrics against defined limits
- **Alert Generation**: Creates alerts for abnormal conditions
- **Severity Classification**: Categorizes issues by impact level

### 6. Recovery & Cleanup
```python
def check_incident_resolution(self):
    if (self.current_incident and 
        datetime.utcnow() >= self.current_incident["end_time"]):
        logger.info(f"✅ INCIDENT RESOLVED: {self.current_incident['description']}")
        self.current_incident = None
```
- **Time-based Resolution**: Automatically ends incidents after duration
- **Resource Cleanup**: Releases any allocated resources
- **Post-mortem**: Logs incident details for analysis

## 📊 Simulation Architecture

### Component Interactions
```mermaid
sequenceDiagram
    participant SG as Service Generator
    participant IS as Incident Simulator
    participant MA as Metrics Aggregator
    participant AL as Alert Manager
    
    loop Every Second
        SG->>SG: Generate baseline metrics
        IS->>SG: Apply incident effects
        SG->>MA: Send updated metrics
        MA->>AL: Check thresholds
        AL->>MA: Trigger alerts if needed
    end
```

### Data Flow
1. **Metric Generation**
   - Each service generates its own metrics
   - Adds random noise to simulate real-world variance
   - Applies daily patterns (e.g., day/night cycles)

2. **Incident Application**
   - Modifies metrics based on incident type
   - May affect multiple related metrics (e.g., CPU ↑ → Latency ↑)
   - Can trigger cascading failures

3. **Alert Processing**
   - Evaluates metrics against thresholds
   - Groups related alerts
   - Applies alert deduplication

4. **State Management**
   - Tracks incident lifecycle
   - Maintains service health status
   - Records historical metrics

```mermaid
graph LR
    A[Service Generators] -->|Generate Metrics| B[Incident Simulator]
    B -->|Apply Incidents| C[Metrics Aggregator]
    C --> D[Console Output]
    C --> E[(Metrics Storage)]
    
    subgraph Services
    S1[API Gateway]
    S2[User Service]
    S3[Order Service]
    S4[Payment Service]
    end
    
    S1 --> S2
    S1 --> S3
    S3 --> S4
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- Docker (optional)
- Kubernetes cluster (for full functionality)
- Cloud provider credentials (AWS/Azure/GCP, if using cloud services)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/dependency-ops-sentinel.git
   cd dependency-ops-sentinel
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. Copy the example configuration file:
   ```bash
   cp config.example.yaml config.yaml
   ```

2. Update `config.yaml` with your settings (see [Configuration Guide](docs/configuration.md))

### Running the Application

#### Development Mode

Start the API server with hot-reload:
```bash
uvicorn dos.api.app:app --reload
```

Access the API at `http://localhost:8000`

#### Using Docker

Build and run with Docker Compose:
```bash
docker-compose up --build
```

This will start:
- DOS API (port 8000)
- Prometheus (port 9090)
- Grafana (port 3000)
- Redis (port 6379)

## 🛠️ Features in Detail

### 1. Real-time Monitoring
- Track system metrics across all layers of your stack
- Visualize service dependencies and health status
- Set custom thresholds and alerts

### 2. Incident Detection
- Machine learning-powered anomaly detection
- Automatic root cause analysis
- Impact assessment and severity scoring

### 3. Automated Remediation
- Suggested actions for common issues
- Integration with CI/CD pipelines
- Rollback capabilities

### 4. Alerting & Notifications
- Configurable alert rules
- Multiple notification channels (Email, Slack, PagerDuty)
- Alert aggregation and deduplication

## 🧠 Technical Deep Dive

### Graph Neural Networks (GNNs)

#### Capturing Service Dependencies
```python
class DependencyGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = GATv2Conv(hidden_dim*4, hidden_dim, heads=4, concat=False)
        self.lin = nn.Linear(hidden_dim, output_dim)
```
- **Message Passing**: Nodes aggregate features from neighbors
- **Attention Mechanism**: Weights connections by importance
- **Hierarchical Learning**: Captures dependencies at multiple levels

### LSTM with Attention for Time-Series

#### Long-Term Pattern Learning
```python
class TemporalAnalyzer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = Attention(hidden_dim)
```
- **Memory Cells**: Maintain state across time steps
- **Gating**: Control information flow with input/forget gates
- **Attention**: Focus on relevant time steps

### Real-time Anomaly Detection

#### Dynamic Thresholding
```python
class RealTimeAnomalyDetector:
    def process_new_point(self, point):
        self.window.append(point)
        if len(self.window) > self.window_size:
            self.window.pop(0)
            
        sequence = torch.FloatTensor(self.window).unsqueeze(0)
        reconstruction, _ = self.model(sequence)
        error = F.mse_loss(sequence[0,-1], reconstruction[0])
        return error > (self.threshold * error.std())
```
- **Sliding Window**: Analyzes recent data points
- **Reconstruction Error**: Detects deviations from learned patterns
- **Adaptive Thresholding**: Adjusts to normal behavior variations

## 📚 Documentation

- [API Reference](docs/api.md)
- [Configuration Guide](docs/configuration.md)
- [Deployment Guide](docs/deployment.md)
- [Development Guide](docs/development.md)

## 🛠️ System Components

### 1. Data Ingestion Layer

#### Kubernetes Metrics Collection
```python
class KubernetesIngestor:
    async def collect_metrics(self):
        pods = self.core_v1.list_pod_for_all_namespaces()
        return [{
            'pod': pod.metadata.name,
            'namespace': pod.metadata.namespace,
            'status': pod.status.phase,
            'containers': [{
                'name': c.name,
                'ready': c.ready,
                'restart_count': c.restart_count
            } for c in pod.status.container_statuses]
        } for pod in pods.items]
```
- **Real-time Monitoring**: Tracks pod lifecycle events
- **Resource Metrics**: CPU/Memory usage per container
- **Health Checks**: Liveness/Readiness probe status

### 2. Alerting System

#### Multi-channel Notification
```python
class AlertManager:
    def __init__(self):
        self.channels = {
            'email': EmailNotifier(),
            'slack': SlackNotifier(),
            'pagerduty': PagerDutyNotifier()
        }
        
    async def send_alert(self, alert, channels=None):
        channels = channels or self.channels.keys()
        for channel in channels:
            await self.channels[channel].send(alert)
```
- **Priority-based Routing**: Critical alerts to multiple channels
- **Deduplication**: Prevents alert storms
- **Acknowledgment**: Tracks alert status

### 3. Dashboard & Visualization

#### Real-time Metrics Display
```javascript
// React component example
const ServiceMap = ({ services }) => (
  <NetworkGraph>
    {services.map(service => (
      <Node 
        key={service.id}
        status={service.health}
        metrics={service.metrics}
      />
    ))}
  </NetworkGraph>
);
```
- **Interactive Service Map**: Visualize dependencies
- **Custom Dashboards**: Build with Grafana
- **Historical Trends**: Time-series visualization

### 4. Automated Remediation

#### Self-healing Actions
```python
class RemediationEngine:
    async def execute_action(self, incident):
        if incident.type == 'high_cpu':
            await self.scale_service(incident.service, 'out')
        elif incident.type == 'pod_crashloop':
            await self.restart_pod(incident.pod)
            
    async def rollback_if_needed(self, action):
        if not await self.verify_remediation(action):
            await self.rollback(action)
```
- **Predefined Playbooks**: Common remediation scenarios
- **Safe Rollback**: Automatic if remediation fails
- **Approval Workflows**: For critical actions

### 5. Data Flow Architecture

```mermaid
sequenceDiagram
    participant C as Collector
    participant P as Processor
    participant A as Analyzer
    participant D as Dashboard
    
    C->>P: Raw Metrics
    P->>A: Processed Data
    A->>D: Anomalies & Insights
    D->>A: User Feedback
    A->>P: Tuning Parameters
```
- **Event-driven**: Asynchronous processing
- **Scalable**: Horizontal scaling of components
- **Resilient**: Retry and circuit breaking

## 🧪 Running Tests

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest tests/

# Run with coverage report
pytest --cov=src tests/
```

## 🚀 Deployment

### Kubernetes

1. Apply the Kubernetes manifests:
   ```bash
   kubectl apply -f k8s/
   ```

2. Access the dashboard:
   ```bash
   kubectl port-forward svc/dos-dashboard 8080:80
   ```
   Then visit `http://localhost:8080`

### Cloud Providers

- [AWS EKS Deployment Guide](docs/deployment/aws-eks.md)
- [Azure AKS Deployment Guide](docs/deployment/azure-aks.md)
- [GCP GKE Deployment Guide](docs/deployment/gcp-gke.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to contribute.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with ❤️ by the DOS team
- Inspired by real-world challenges in managing complex distributed systems
- Thanks to all contributors who have helped shape this project

## 📬 Contact

For questions or support, please [open an issue](https://github.com/your-org/dependency-ops-sentinel/issues) or contact our team at support@example.com
