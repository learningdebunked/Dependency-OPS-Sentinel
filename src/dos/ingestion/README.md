# Signal Ingestion Module (Experimental)

This module provides a flexible and extensible framework for collecting, validating, and processing monitoring signals from various sources.

⚠️ **Status**: Experimental - APIs may change in future releases

## Core Concepts

### 1. Signals
Base data structures representing different types of monitoring data:
- `Signal`: Base class for all signals
- `Metric`: Numerical measurements over time (e.g., CPU usage, request rate)
- `Log`: Textual log entries
- `Trace`: Distributed tracing information

### 2. Data Sources
Components that collect signals from various systems:
- Abstract base class: `DataSource`
- Built-in implementations:
  - `PrometheusSource` (Planned)
  - `ElasticsearchSource` (Planned)
  - `KafkaSource` (Planned)

### 3. Processors
Components that transform, filter, or enrich signals:
- Abstract base class: `Processor`
- Built-in processors:
  - `FilterProcessor`: Filter signals based on conditions
  - `TransformProcessor`: Transform signal fields
  - `EnrichProcessor`: Add contextual information

## Usage Example

```python
import asyncio
from datetime import datetime, timedelta
from typing import AsyncGenerator

from dos.ingestion import DataSource, Processor, Metric, SignalType

# 1. Define a custom data source
class RandomMetricSource(DataSource):
    """Generates random metrics for demonstration."""
    
    async def connect(self):
        print("Connected to random metric source")
        
    async def disconnect(self):
        print("Disconnected from random metric source")
        
    async def stream_signals(self) -> AsyncGenerator[Metric, None]:
        while True:
            yield Metric(
                name="cpu.usage",
                value=random.uniform(0, 100),
                unit="percent",
                source="demo",
                labels={"host": "example.com"}
            )
            await asyncio.sleep(1)

# 2. Define a custom processor
class ThresholdAlertProcessor(Processor):
    """Generates alerts when metrics exceed thresholds."""
    
    def __init__(self, threshold: float = 90.0):
        self.threshold = threshold
        
    async def process(self, signal: Metric) -> Optional[Metric]:
        if signal.value > self.threshold:
            print(f"🚨 Alert: {signal.name} = {signal.value}{signal.unit} "
                  f"exceeds threshold {self.threshold}{signal.unit}")
        return signal

# 3. Run the pipeline
async def main():
    source = RandomMetricSource({})
    processor = ThresholdAlertProcessor(threshold=85.0)
    
    async with source:
        async for signal in source.stream_signals():
            await processor(signal)

if __name__ == "__main__":
    asyncio.run(main())
```

## Implementing Custom Components

### 1. Custom Data Source

```python
from dos.ingestion.sources.base import DataSource
from dos.ingestion.schemas import Signal

class CustomSource(DataSource):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your source here
        
    async def connect(self):
        # Connect to your data source
        pass
        
    async def disconnect(self):
        # Clean up resources
        pass
        
    async def stream_signals(self) -> AsyncGenerator[Signal, None]:
        while self.running:
            # Yield signals from your source
            yield Signal(...)
```

### 2. Custom Processor

```python
from dos.ingestion.processors.base import Processor
from dos.ingestion.schemas import Signal

class CustomProcessor(Processor):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your processor
        
    async def process(self, signal: Signal) -> Optional[Signal]:
        # Process the signal
        return signal  # Return None to filter out
```

## Status and Roadmap

### Implemented
- Base signal models (Metric, Log, Trace)
- Basic data source and processor interfaces
- Example implementations

### Planned
- Built-in data source implementations
- More processor types
- Batch processing support
- Schema validation and evolution
- Performance optimizations

## Contributing

Contributions are welcome! Please see the main project's contributing guidelines for details.
