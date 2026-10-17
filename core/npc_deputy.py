"""Backward-compatible import shim.

Deprecated module name: npc_deputy
Preferred module name: tas
"""

from .tas import TASRoles, TripartiteAuditorSystem, NPCDeputyManager

__all__ = ["TASRoles", "TripartiteAuditorSystem", "NPCDeputyManager"]
