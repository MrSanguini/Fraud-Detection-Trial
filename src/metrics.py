"""Evaluation harness: ranking metrics, threshold metrics, and a money cost.

Cost model (asymmetric, amount-aware):
  - false negative (missed fraud): you eat the transaction's actual amount
  - false positive (declined legit): a fixed friction cost per event (parameter)
Reported per 1,000,000 transactions so configurations compare on one scale."""

import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix)

def ranking_metrics(y_true, y_scores):
    return {"AUROC":  roc_auc_score(y_true, y_scores),
            "PR_AUC": average_precision_score(y_true, y_scores)}

def threshold_metrics(y_true, y_scores, threshold):
    y_pred = (np.asarray(y_scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"threshold": threshold,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

def cost(y_true, y_scores, amounts, threshold, fp_unit_cost):
    y_true  = np.asarray(y_true)
    amounts = np.nan_to_num(np.asarray(amounts, dtype=float))
    y_pred  = (np.asarray(y_scores) >= threshold).astype(int)
    fn_mask = (y_true == 1) & (y_pred == 0)        # missed fraud
    fp_mask = (y_true == 0) & (y_pred == 1)        # declined legit
    fn_cost = amounts[fn_mask].sum()
    fp_cost = fp_mask.sum() * fp_unit_cost
    total   = fn_cost + fp_cost
    return {"fn_cost": fn_cost, "fp_cost": fp_cost, "total_cost": total,
            "cost_per_million": total / len(y_true) * 1_000_000}

def best_threshold(y_true, y_scores, amounts, fp_unit_cost, grid=None):
    """Sweep thresholds; return the one minimizing total money cost."""
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    costs = [(t, cost(y_true, y_scores, amounts, t, fp_unit_cost)["total_cost"])
             for t in grid]
    t_best, c_best = min(costs, key=lambda x: x[1])
    return t_best, c_best, costs
