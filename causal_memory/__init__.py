"""
Causal Memory Layer - A causal graph-based memory system for AI agents.

Built on HydraDB for the Hack Hydra hackathon.
"""

from .client import HydraDBClient
from .memory import CausalMemory
from .schema import Event, CausalRelation

__version__ = "0.1.0"
__all__ = ["CausalMemory", "HydraDBClient", "Event", "CausalRelation"]
