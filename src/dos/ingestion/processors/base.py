"""
Base classes for signal processors (Experimental)

This module defines the abstract base class for all signal processors.
"""
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, Dict, Any
from ..schemas import Signal

class Processor(ABC):
    """
    Abstract base class for all signal processors.
    
    Implement this class to add custom processing logic.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the processor with optional configuration."""
        self.config = config or {}
    
    @abstractmethod
    async def process(self, signal: Signal) -> Optional[Signal]:
        """
        Process a single signal.
        
        Args:
            signal: Input signal to process
            
        Returns:
            Processed signal or None if signal should be filtered out
        """
        pass
    
    async def __call__(self, signal: Signal) -> Optional[Signal]:
        """Make the processor callable."""
        return await self.process(signal)
