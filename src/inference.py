"""Inference seam. The ONLY entry point the UI/API is allowed to call."""
from dataclasses import dataclass, field

@dataclass
class ScoreResult:
    probability: float
    decision: str                 # "approve" | "review" | "block"
    rules_fired: list = field(default_factory=list)

def score(transaction: dict) -> ScoreResult:
    """Run preprocess -> model -> rules for ONE transaction.
    Implemented across Phases 2, 4, and 6. Stub for now."""
    raise NotImplementedError("Filled in by Phase 6.")
