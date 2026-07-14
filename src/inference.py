"""Inference seam: preprocess -> model -> rules. The ONLY entry point for UI/API."""

from dataclasses import dataclass, field
from typing import List

@dataclass
class ScoreResult:
    probability: float
    decision: str                       # "approve" | "review" | "block"
    rules_fired: List[str] = field(default_factory=list)

# ---- Tunable rule parameters -------------------------------------------------
REVIEW_BAND      = 0.35    # >= this -> review
BLOCK_BAND       = 0.85    # >= this -> block
HIGH_AMOUNT      = 1000.0  # guardrail: large transactions get extra scrutiny
GUARDRAIL_SCORE  = 0.15    # ...but only if the model shows *some* suspicion
TRUSTED_MAX_AMT  = 25.0    # guardrail: tiny amounts are cheap to let through
# -----------------------------------------------------------------------------

def apply_rules(prob: float, txn: dict) -> ScoreResult:
    """Outer decision layer. Takes the model's calibrated probability plus the
    raw transaction, returns a decision. Never modifies the model."""
    fired = []
    amt = float(txn.get("TransactionAmt") or 0.0)

    # 1) SCORE BANDS (the model's own verdict)
    if   prob >= BLOCK_BAND:  decision = "block";   fired.append("band:block")
    elif prob >= REVIEW_BAND: decision = "review";  fired.append("band:review")
    else:                     decision = "approve"

    # 2) GUARDRAIL — escalate: large amount + non-trivial suspicion
    if amt >= HIGH_AMOUNT and prob >= GUARDRAIL_SCORE and decision == "approve":
        decision = "review"; fired.append("guardrail:high_amount_escalate")

    # 3) GUARDRAIL — de-escalate: trivially small amounts not worth blocking
    if amt <= TRUSTED_MAX_AMT and decision == "block":
        decision = "review"; fired.append("guardrail:low_amount_deescalate")

    # 4) COMPLIANCE OVERRIDE — always wins, regardless of the model
    if txn.get("compliance_flag"):
        decision = "block"; fired.append("compliance:hard_block")

    return ScoreResult(probability=float(prob), decision=decision, rules_fired=fired)
