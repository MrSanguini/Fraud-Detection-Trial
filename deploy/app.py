"""Fraud detection API. Standalone service; the demo UI is just one client."""
import json, time
from pathlib import Path
from typing import List, Optional

import joblib, numpy as np, pandas as pd, xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import inference, rules_config as cfg
from inference import apply_rules, top_risk_factors

HERE = Path(__file__).parent

# ---- Load once at startup ---------------------------------------------------
booster = xgb.Booster(); booster.load_model(str(HERE / "final_xgb.json"))
calibrator = joblib.load(HERE / "calibrator.joblib")
preprocessor = joblib.load(HERE / "preprocessor.joblib")
FEATURE_ORDER = booster.feature_names

samples = pd.read_parquet(HERE / "samples.parquet")
samples_meta = json.load(open(HERE / "samples_meta.json"))
cost_grid = json.load(open(HERE / "cost_grid.json"))
triage = json.load(open(HERE / "triage.json"))
T_STAR = triage["t_star"]

app = FastAPI(
    title="Fraud Detection API",
    description="Calibrated XGBoost scoring with a rules-based decision layer.",
    version="1.0.0",
)

# ---- Contracts --------------------------------------------------------------
class RiskFactor(BaseModel):
    label: str
    value: Optional[str] = None

class ScoreRequest(BaseModel):
    TransactionAmt: float = Field(..., examples=[249.99])
    ProductCD: Optional[str] = Field(None, examples=["W"])
    card4: Optional[str] = Field(None, examples=["visa"])
    card6: Optional[str] = Field(None, examples=["debit"])
    P_emaildomain: Optional[str] = Field(None, examples=["gmail.com"])
    R_emaildomain: Optional[str] = None
    DeviceType: Optional[str] = Field(None, examples=["mobile"])
    id_30: Optional[str] = None       # operating system
    id_31: Optional[str] = None       # browser
    sample_id: Optional[int] = Field(None, description="Score a stored sample row instead")

class ScoreResponse(BaseModel):
    probability: float
    risk_band: str
    decision: str
    rules_fired: List[str]
    rule_descriptions: List[str]
    risk_factors: List[RiskFactor]
    behavioral_factors: List[RiskFactor]
    latency_ms: float

def band(p: float) -> str:
    return ("Critical" if p >= 0.85 else "High" if p >= 0.50
            else "Elevated" if p >= T_STAR else "Low")

# ---- Endpoints --------------------------------------------------------------
@app.post("/score", response_model=ScoreResponse, tags=["scoring"])
def score(req: ScoreRequest):
    """Score one transaction. Returns the calibrated fraud probability, the
    decision after the rules layer, and the top interpretable risk factors."""
    t0 = time.perf_counter()

    if req.sample_id is not None:
        if not 0 <= req.sample_id < len(samples):
            raise HTTPException(404, f"sample_id must be 0..{len(samples)-1}")
        row = samples.iloc[[req.sample_id]]
    else:
        # Build a full feature row: start from a template, overwrite what's given
        row = samples.iloc[[0]].copy()
        for k, v in req.model_dump(exclude_none=True).items():
            if k in row.columns:
                if str(row[k].dtype) == "category":
                    if v not in row[k].cat.categories:
                        row[k] = row[k].cat.add_categories([v])
                row.loc[row.index[0], k] = v

    raw = booster.predict(xgb.DMatrix(row[FEATURE_ORDER], enable_categorical=True))
    prob = float(calibrator.predict(raw)[0])

    txn = {c: (None if pd.isna(row[c].iloc[0]) else row[c].iloc[0])
           for c in row.columns}
    res = apply_rules(prob, txn, T_STAR)
    factors = top_risk_factors(booster, row, FEATURE_ORDER, txn, k=3)
    factors_interp = top_risk_factors(booster, row, FEATURE_ORDER, txn, k=3, interpretable_only=True)
    factors_raw    = top_risk_factors(booster, row, FEATURE_ORDER, txn, k=3, interpretable_only=False)

    return ScoreResponse(
        probability=prob, risk_band=band(prob), decision=res.decision,
        rules_fired=res.rules_fired,
        rule_descriptions=[cfg.RULE_DESCRIPTIONS.get(r, r) for r in res.rules_fired],
        risk_factors=[RiskFactor(label=f["label"],
                                 value=None if f["value"] is None else str(f["value"]))
                      for f in factors_interp],
        behavioral_factors = [RiskFactor(label=f["label"],
                                 value=None if f["value"] is None else str(f["value"]))
                      for f in factors_raw],
        latency_ms=(time.perf_counter() - t0) * 1000,
    )

@app.get("/comparison", tags=["metrics"])
def comparison(fp_cost: int = 25):
    """Rules-only vs model, at a given cost per wrongly-declined customer."""
    key = min(cost_grid["fp_costs"], key=lambda c: abs(c - fp_cost))
    return {"fp_cost": key, **cost_grid["grid"][str(key)]}

@app.get("/triage", tags=["metrics"])
def get_triage():
    """How the system routes traffic: approve / review / block."""
    return triage

@app.get("/samples", tags=["scoring"])
def list_samples():
    """Stored example transactions available for scoring."""
    return {"count": len(samples),
            "fields": [c for c in ["TransactionAmt", "card4", "card6",
                                   "P_emaildomain", "DeviceType"] if c in samples.columns]}

@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "threshold": T_STAR, "n_features": len(FEATURE_ORDER)}

# ---- Static frontend (a client of the API above, not part of it) ------------
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(HERE / "static" / "index.html")