"""
Dependency Ops Sentinel - Signal Ingestion Module (Experimental)

This module handles the collection, validation, and processing of monitoring signals
from various sources. It provides a pluggable architecture for adding new data sources
and processing pipelines.

⚠️ Status: Experimental - API may change in future releases
"""

__version__ = "0.1.0"
__status__ = "experimental"

from .sources.base import DataSource
from .processors.base import Processor
from .schemas import Signal, Metric, Log, Trace

__all__ = [
    'DataSource',
    'Processor',
    'Signal', 
    'Metric',
    'Log',
    'Trace'
]
