"""Inference seam: preprocess -> model -> rules. The ONLY entry point for UI/API."""
import rules_config as cfg
from dataclasses import dataclass, field
from typing import List

@dataclass
class ScoreResult:
    probability: float
    decision: str                       # "approve" | "review" | "block"
    rules_fired: List[str] = field(default_factory=list)
    risk_factors: List[dict] = field(default_factory=list)   # NEW

# ---- Tunable rule parameters (Defaults) --------------------------------------
BLOCK_BAND       = 0.85    # >= this -> block
HIGH_AMOUNT      = 1000.0  # guardrail: large transactions get extra scrutiny
GUARDRAIL_SCORE  = 0.15    # ...but only if the model shows *some* suspicion
TRUSTED_MAX_AMT  = 25.0    # guardrail: tiny amounts are cheap to let through
# -----------------------------------------------------------------------------

def apply_rules(prob: float, txn: dict, review_threshold: float) -> ScoreResult:
    """Outer decision layer. Takes the model's calibrated probability plus the
    raw transaction, returns a decision. Never modifies the model."""
    fired = []
    amt = float(txn.get("TransactionAmt") or 0.0)

    # 1) SCORE BANDS (the model's own verdict)
    if   prob >= BLOCK_BAND:       decision = "block";   fired.append("band:block")
    elif prob >= review_threshold: decision = "review";  fired.append("band:review")
    else:                          decision = "approve"

    # 2) GUARDRAIL — escalate: large amount + non-trivial suspicion
    if amt >= HIGH_AMOUNT and prob >= GUARDRAIL_SCORE and decision == "approve":
        decision = "review"; fired.append("guardrail:high_amount_escalate")

    # 3) GUARDRAIL — de-escalate: trivially small amounts not worth blocking
    if amt <= TRUSTED_MAX_AMT and decision == "block":
        decision = "review"; fired.append("guardrail:low_amount_deescalate")

    # 4) COMPLIANCE OVERRIDES — always win, regardless of the model's opinion
    email   = txn.get("P_emaildomain")
    r_email = txn.get("R_emaildomain")
    region  = txn.get("addr1")
    card    = txn.get("card6")

    # 4a) High-risk email domain on a large transfer
    if email in cfg.HIGH_RISK_EMAIL_DOMAINS and amt >= cfg.HIGH_RISK_EMAIL_AMOUNT:
        decision = "block"; fired.append("compliance:high_risk_email")

    # 4b) Sanctioned region (masked codes in this dataset; real names in production)
    if region is not None and float(region) in cfg.SANCTIONED_REGIONS.values():
        decision = "block"; fired.append("compliance:sanctioned_region")

    # 4c) Restricted card type above its permitted limit
    if card in cfg.RESTRICTED_CARD_TYPES and amt >= cfg.RESTRICTED_CARD_AMOUNT:
        decision = "block"; fired.append("compliance:restricted_card")

    # 4d) Purchaser/recipient email mismatch on a large transfer -> human review
    if (email and r_email and email != r_email
            and amt >= cfg.EMAIL_MISMATCH_AMOUNT and decision == "approve"):
        decision = "review"; fired.append("compliance:email_mismatch")

    return ScoreResult(probability=float(prob), decision=decision, rules_fired=fired)

import numpy as np

# --- Explanation layer -------------------------------------------------------
# Human-readable names for the model's features. Anything not listed falls back
# to the raw column name.
FEATURE_LABELS = {
    "TransactionAmt":  "Transaction amount",
    "cents":           "Amount pattern (round/odd cents)",
    "card1":           "Card identifier",
    "card2":           "Card issuer code",
    "card4":           "Card network",
    "card6":           "Card type",
    "addr1":           "Billing region",
    "addr1_FE":        "Billing region rarity",
    "card1_FE":        "Card rarity",
    "P_emaildomain":   "Purchaser email domain",
    "P_emaildomain_FE":"Email domain rarity",
    "R_emaildomain":   "Recipient email domain",
    "DeviceType":      "Device type",
    "DeviceInfo":      "Device details",
    "id_30":           "Operating system",
    "id_31":           "Browser",
    "dist1":           "Billing/shipping distance",
    "hour":            "Time of day",
    "weekday":         "Day of week",
    "ProductCD":       "Product category",
}

def label_for(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    if feature.endswith("_FE"):                      # frequency-encoded columns
        base = feature[:-3]
        return f"{FEATURE_LABELS.get(base, base)} rarity"
    if feature.startswith(("V", "C", "D", "M", "id_")):
        return f"Behavioral signal {feature}"        # the anonymized Vesta features
    return feature

# Feature prefixes that are anonymized in this dataset (no business meaning)
OPAQUE_PREFIXES = ("V", "C", "D", "M", "id_1", "id_2", "id_3")

def is_interpretable(feature: str) -> bool:
    """True if a human could actually read this feature's name and value."""
    if feature in FEATURE_LABELS:
        return True
    if feature.endswith("_FE"):
        return feature[:-3] in FEATURE_LABELS      # e.g. P_emaildomain_FE -> yes
    return not feature.startswith(OPAQUE_PREFIXES)

def top_risk_factors(booster, row_df, feature_order, txn: dict, k: int = 3,
                     interpretable_only: bool = True):
    """Top-k risk-INCREASING features.

    interpretable_only=True filters to features with business meaning. The model
    still uses the anonymized signals — they're just not shown, because in this
    public dataset they have no readable meaning. Set False to see the raw top-k."""
    import xgboost as xgb
    dm = xgb.DMatrix(row_df[feature_order], enable_categorical=True)
    contribs = booster.predict(dm, pred_contribs=True)[0]
    values = contribs[:-1]

    idx = np.argsort(values)[::-1]                        # most risk-increasing first
    out, opaque_weight = [], 0.0
    for i in idx:
        if values[i] <= 0:
            break                                          # sorted: no positives left
        feat = feature_order[i]
        if interpretable_only and not is_interpretable(feat):
            opaque_weight += float(values[i])              # track what we're hiding
            continue
        out.append({"feature": feat, "label": label_for(feat),
                    "value": txn.get(feat), "impact": float(values[i])})
        if len(out) == k:
            break

    # Honest bookkeeping: how much risk came from signals we didn't show
    for f in out:
        f["hidden_signal_weight"] = opaque_weight
    return out

def explain(result: ScoreResult) -> str:
    """One-paragraph, non-technical explanation of a decision."""
    band = ("Critical" if result.probability >= 0.85 else
            "High"     if result.probability >= 0.50 else
            "Elevated" if result.probability >= 0.15 else "Low")

    lines = [f"{result.decision.upper()} — {band} risk ({result.probability:.1%})"]

    if result.risk_factors:
        lines.append("Top interpretable risk factors:")
        for i, f in enumerate(result.risk_factors, 1):
            val = f["value"]
            val_str = f" — {val}" if val is not None else ""
            lines.append(f"  {i}. {f['label']}{val_str}")
        hidden = result.risk_factors[0].get("hidden_signal_weight", 0)
        if hidden > 0:
            lines.append("  (Additional behavioral signals also contributed to this score.)")

    compliance = [r for r in result.rules_fired if r.startswith("compliance:")]
    if compliance:
        lines.append("Compliance rules triggered:")
        for r in compliance:
            lines.append(f"  • {cfg.RULE_DESCRIPTIONS.get(r, r)}")
    return "\n".join(lines)
