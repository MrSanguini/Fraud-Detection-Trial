"""Leakage-safe preprocessing for the fraud model (lean feature set).

Fit on TRAIN only; transform train, test, and live transactions identically.
Trees: no scaling, no imputation. Adds three cheap engineered features:
cents, D-column normalization, and train-fit frequency encoding."""

import numpy as np
import pandas as pd

ID_COLS  = ["TransactionID"]
TIME_COL = "TransactionDT"
TARGET   = "isFraud"

# Categoricals to frequency-encode (high-cardinality, rarity is a fraud signal)
FE_COLS = ["addr1", "card1", "card2", "card3", "P_emaildomain"]
# D columns that drift with time and benefit from normalization
D_NORM  = [4, 6, 10, 11, 12, 13, 14, 15]

def add_stateless_features(df):
    """Per-row features with NO learned state: safe on any single row."""
    out = df.copy()
    # 1) cents: fractional part of the amount (round/odd amounts flag fraud)
    if "TransactionAmt" in out.columns:
        amt = out["TransactionAmt"]
        out["cents"] = (amt - np.floor(amt)).astype("float32")
    # 2) D-column normalization: anchor drifting time-deltas to a fixed point
    if TIME_COL in out.columns:
        day = out[TIME_COL] / np.float32(24 * 60 * 60)
        for i in D_NORM:
            col = f"D{i}"
            if col in out.columns:
                out[col] = (out[col] - day).astype("float32")
    # 3) simple cyclical time features (from Phase 2)
    if TIME_COL in out.columns:
        secs = out[TIME_COL]
        out["hour"]    = (secs // 3600 % 24).astype("int16")
        out["weekday"] = (secs // (3600 * 24) % 7).astype("int16")
    return out

class Preprocessor:
    def __init__(self):
        self.feature_cols = None
        self.cat_levels = {}
        self.freq_maps = {}          # col -> {value: frequency}, learned on TRAIN

    def fit(self, train_df):
        df = add_stateless_features(train_df)
        # Learn frequency maps from TRAIN only
        for col in FE_COLS:
            if col in df.columns:
                self.freq_maps[col] = (
                    df[col].value_counts(normalize=True, dropna=True).to_dict()
                )
        df = self._apply_freq(df)
        drop = set(ID_COLS + [TIME_COL, TARGET])
        self.feature_cols = [c for c in df.columns if c not in drop]
        for c in self.feature_cols:
            if str(df[c].dtype) == "category":
                self.cat_levels[c] = df[c].cat.categories
        return self

    def _apply_freq(self, df):
        for col, vc in self.freq_maps.items():
            if col in df.columns:
                df[col + "_FE"] = df[col].map(vc).fillna(-1).astype("float32")
        return df

    def transform(self, df):
        df = add_stateless_features(df)
        df = self._apply_freq(df)               # uses TRAIN-learned maps
        X = pd.DataFrame(index=df.index)
        for c in self.feature_cols:
            col = df[c] if c in df.columns else pd.Series(pd.NA, index=df.index)
            if c in self.cat_levels:
                col = pd.Categorical(col, categories=self.cat_levels[c])
            X[c] = col
        return X[self.feature_cols]
