"""
Base classes for data sources (Experimental)

This module defines the abstract base class for all data sources.
"""
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, Dict, Any
from ..schemas import Signal

class DataSource(ABC):
    """
    Abstract base class for all data sources.
    
    Implement this class to add support for new data sources.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data source with configuration."""
        self.config = config
        self.running = False
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the data source."""
        pass
    
    @abstractmethod
    async def stream_signals(self) -> AsyncGenerator[Signal, None]:
        """Stream signals from the data source."""
        if False:  # This makes the method a generator
            yield None
    
    async def __aenter__(self):
        """Context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.disconnect()
