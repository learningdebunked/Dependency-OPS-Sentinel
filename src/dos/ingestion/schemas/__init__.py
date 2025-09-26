""
Signal schemas for data ingestion (Experimental)

This module defines the core data models used throughout the ingestion pipeline.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Union, TypeVar, Generic
from pydantic import BaseModel, Field, validator

T = TypeVar('T')

class SignalType(str, Enum):
    """Types of signals supported by the system."""
    METRIC = "metric"
    LOG = "log"
    TRACE = "trace"
    EVENT = "event"

class Signal(BaseModel):
    """
    Base class for all signals in the system.
    
    Attributes:
        signal_type: Type of the signal (metric, log, trace, etc.)
        source: Source system identifier
        timestamp: When the signal was generated
        labels: Key-value pairs for categorization
        raw: Original raw data (if applicable)
    """
    signal_type: SignalType
    source: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = Field(default_factory=dict)
    raw: Optional[Dict[str, Any]] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class Metric(Signal):
    """
    Metric signal representing numerical measurements over time.
    
    Attributes:
        name: Metric name (e.g., 'cpu.usage')
        value: Numerical value of the metric
        unit: Unit of measurement (e.g., 'percent', 'bytes')
    """
    signal_type: SignalType = SignalType.METRIC
    name: str
    value: Union[float, int]
    unit: Optional[str] = None

class Log(Signal):
    """
    Log signal representing textual log entries.
    
    Attributes:
        message: Log message content
        level: Log level (e.g., 'info', 'error')
        logger: Name of the logger that produced this log
    """
    signal_type: SignalType = SignalType.LOG
    message: str
    level: str
    logger: Optional[str] = None

class Trace(Signal):
    """
    Distributed trace signal for request flow tracking.
    
    Attributes:
        trace_id: Unique identifier for the trace
        span_id: Unique identifier for this span
        parent_span_id: Identifier of the parent span (if any)
        operation_name: Name of the operation being traced
        duration_ms: Duration of the operation in milliseconds
    """
    signal_type: SignalType = SignalType.TRACE
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation_name: str
    duration_ms: float

# Add these to __all__ in the module's __init__.py
__all__ = ['Signal', 'Metric', 'Log', 'Trace', 'SignalType']
