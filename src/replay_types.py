"""
Shared Types and Enums
======================

Common types used across the replay system to avoid circular dependencies.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class ReplayDecision(Enum):
    """Decision made by the replay engine."""
    REPLAY = "replay"  # Found similar state, replay action
    AI_FALLBACK = "ai_fallback"  # Novel state, needs AI
    NO_ACTION = "no_action"  # Matched state has no action


@dataclass
class ReplayResult:
    """Result from the replay engine."""
    decision: ReplayDecision
    confidence: float  # Similarity score (0-1)
    action_text: Optional[str]  # Action to replay (if REPLAY)
    think_text: Optional[str]  # Reasoning hint (if available)
    matched_node_id: Optional[int]  # ID of matched node (if any)
    
    def should_replay(self) -> bool:
        """Check if this result indicates we should replay."""
        return self.decision == ReplayDecision.REPLAY

