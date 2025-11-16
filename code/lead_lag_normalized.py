import os
import numpy as np
import pandas as pd

# Reuse plotting from the existing analysis
from lead_lag_analysis import plot_event

REPO_ROOT = "/Users/beszabo/bene/topicality-online"
DERIVED_DIR = os.path.join(REPO_ROOT, "data", "derived")
FIG_DIR = os.path.join(REPO_ROOT, "figures")
PANEL_CSV = os.path.join(DERIVED_DIR, "company_weekly_panel.csv")

os.makedirs(FIG_DIR, exist_ok=True)


def add_normalizations(panel: pd.DataFrame, monthly: bool = False) -> pd.DataFrame:
    if monthly:
        panel = panel.sort_values(["company", "month_start"]).copy()
    else:
        panel = panel.sort_values(["company", "week_start"]).copy()
    # z-score per company
    mu = panel.groupby("company")["num_memes"].transform("mean")
    sd = panel.groupby("company")["num_memes"].transform("std").replace(0, np.nan)
    panel["num_memes_z"] = (panel["num_memes"] - mu) / sd
    panel["num_memes_z"] = panel["num_memes_z"].fillna(0.0)
    # ratio to rolling baseline (previous 8 weeks)
    def _rolling_mean(s: pd.Series) -> pd.Series:
        if monthly:
            return s.shift(1).rolling(window=4, min_periods=3).mean()
        else:
            return s.shift(1).rolling(window=8, min_periods=3).mean()
    roll = panel.groupby("company")["num_memes"].apply(_rolling_mean).reset_index(level=0, drop=True)
    panel["num_memes_rel"] = panel["num_memes"] / (roll.replace(0, np.nan))
    panel["num_memes_rel"] = panel["num_memes_rel"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return panel


def event_study_value(panel: pd.DataFrame, pos_feature: str, neg_feature: str, value_col: str, window: int = 3,
                      pos_q: float = 0.90, neg_q: float = 0.90, monthly: bool = False):
    rows_pos, rows_neg = [], []
    for company, g in panel.groupby('company'):
        if monthly:
            g = g.sort_values('month_start').reset_index(drop=True)
        else:
            g = g.sort_values('week_start').reset_index(drop=True)
        if g.empty:
            continue
        pos_thresh = g[pos_feature].quantile(pos_q)
        neg_thresh = g[neg_feature].quantile(neg_q)
        n = len(g)
        for i in range(n):
            if pd.notna(g.loc[i, pos_feature]) and g.loc[i, pos_feature] >= pos_thresh:
                for tau in range(-window, window+1):
                    j = i + tau
                    if 0 <= j < n and pd.notna(g.loc[j, value_col]):
                        rows_pos.append({'company': company, 'tau': tau, value_col: float(g.loc[j, value_col])})
            if pd.notna(g.loc[i, neg_feature]) and g.loc[i, neg_feature] >= neg_thresh:
                for tau in range(-window, window+1):
                    j = i + tau
                    if 0 <= j < n and pd.notna(g.loc[j, value_col]):
                        rows_neg.append({'company': company, 'tau': tau, value_col: float(g.loc[j, value_col])})
    return pd.DataFrame(rows_pos), pd.DataFrame(rows_neg)