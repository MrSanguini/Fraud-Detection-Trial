"""Business-editable rule configuration.

Rules are written in business language here. Non-engineers should be able to
edit THIS file without touching inference logic.

NOTE ON MASKED FIELDS: In this public dataset, geography (addr1/addr2) is
anonymized to numeric codes. In production these become real country/region
names and the mapping below disappears entirely."""

# --- Email domains treated as elevated-risk for large transfers -------------
HIGH_RISK_EMAIL_DOMAINS = {
    "protonmail.com",
    "mail.com",
    "outlook.es",
}
HIGH_RISK_EMAIL_AMOUNT = 500.0     # only applies above this amount

# --- Sanctioned regions -----------------------------------------------------
# Keys are human labels; values are this dataset's masked addr1 codes.
# In production: {"IRAN": "IR", "NORTH_KOREA": "KP", ...}
SANCTIONED_REGIONS = {
    "REGION_A": 325.0,
    "REGION_B": 337.0,
}

# --- Card types barred from high-value transfers ----------------------------
RESTRICTED_CARD_TYPES  = {"charge card"}
RESTRICTED_CARD_AMOUNT = 2000.0

# --- Mismatched purchaser/recipient email on large amounts ------------------
EMAIL_MISMATCH_AMOUNT = 1000.0

# --- Human-readable names for every rule (for the UI) -----------------------
RULE_DESCRIPTIONS = {
    "band:block":   "Model risk score in auto-block range",
    "band:review":  "Model risk score in review range",
    "guardrail:high_amount_escalate":  "Large transaction with elevated risk — escalated to review",
    "guardrail:low_amount_deescalate": "Low-value transaction — downgraded from block to review",
    "compliance:high_risk_email":      "High-risk email domain on a large transfer",
    "compliance:sanctioned_region":    "Transaction originates in a sanctioned region",
    "compliance:restricted_card":      "Restricted card type used above permitted limit",
    "compliance:email_mismatch":       "Purchaser and recipient email domains differ on a large transfer",
}
