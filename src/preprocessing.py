"""Leakage-safe preprocessing for the fraud model.

Fit on TRAIN only; transform train, test, and live transactions identically.
Built for gradient-boosted trees: no scaling, no imputation (the booster
handles NaNs and is scale-invariant)."""

import pandas as pd

ID_COLS  = ["TransactionID"]
TIME_COL = "TransactionDT"
TARGET   = "isFraud"

def add_time_features(df):
    """Per-row cyclical time signals from the DT offset. Safe: no cross-row
    or future information is used."""
    out = df.copy()
    if TIME_COL in out.columns:
        secs = out[TIME_COL]
        out["hour"]    = (secs // 3600 % 24).astype("int16")
        out["weekday"] = (secs // (3600 * 24) % 7).astype("int16")
    return out

class Preprocessor:
    def __init__(self):
        self.feature_cols = None
        self.cat_levels = {}          # col -> categories learned from TRAIN

    def fit(self, train_df):
        df = add_time_features(train_df)
        drop = set(ID_COLS + [TIME_COL, TARGET])
        self.feature_cols = [c for c in df.columns if c not in drop]
        for c in self.feature_cols:
            if str(df[c].dtype) == "category":
                self.cat_levels[c] = df[c].cat.categories
        return self

    def transform(self, df):
        df = add_time_features(df)
        
        # Collect columns in a dict to avoid fragmentation
        cols_to_concat = {}
        for c in self.feature_cols:
            col = df[c] if c in df.columns else pd.Series(pd.NA, index=df.index)
            if c in self.cat_levels:                 # re-apply TRAIN categories;
                col = pd.Categorical(col,            # unseen values -> NaN
                                     categories=self.cat_levels[c])
            cols_to_concat[c] = col
        
        # Create final DataFrame in one go
        X = pd.DataFrame(cols_to_concat, index=df.index)
        return X[self.feature_cols]
