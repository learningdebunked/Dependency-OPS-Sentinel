# Hello World Example for DOS

This minimal example demonstrates the core functionality of Dependency Ops Sentinel (DOS) with synthetic data.

## Features

- ✅ Basic service monitoring
- ✅ Synthetic metric generation
- ✅ Alerting based on thresholds
- ✅ Simple command-line output
- ✅ Configurable settings

## Prerequisites

- Python 3.8+
- PyYAML (`pip install pyyaml`)

## Quick Start

1. Navigate to the example directory:
   ```bash
   cd examples/hello_world
   ```

2. Run the example:
   ```bash
   python hello_dos.py
   ```

3. You'll see output like this:
   ```
   🚀 DOS Hello World initialized!
   Monitoring service: hello-service

   📊 Starting monitoring for 30 seconds...

   [14:30:45] CPU:  25.3% | MEM: 312.4MB | LAT:  87.2ms | REQ: 1
   [14:30:46] CPU:  72.5% | MEM: 298.1MB | LAT: 134.7ms | REQ: 2
      🚨 ALERT: High CPU usage: 72.5%
   [14:30:47] CPU:  18.7% | MEM: 287.6MB | LAT:  92.3ms | REQ: 3
   ```

## Configuration

Edit `config.yaml` to adjust:
- Monitoring intervals
- Alert thresholds
- Service settings
- Logging levels

## Extending the Example

To add more complex monitoring:
1. Add new metrics to the `generate_metrics` method
2. Update the alerting rules in `config.yaml`
3. Add new alert checks in the `check_alerts` method

## Next Steps

- Connect to real services by updating the metric collection
- Add more sophisticated alerting rules
- Integrate with external monitoring systems
- Set up automated responses to common issues
