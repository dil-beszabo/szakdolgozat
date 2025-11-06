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


def run():
    # Z-score event study
    pos_z, neg_z = event_study_value(panel, pos_feature='mean_pos', neg_feature='mean_neg', value_col='num_memes_z', window=3)
    plot_event(pos_z, 'num_memes_z', 'Event: Positive news vs normalized (z) meme volume', os.path.join(FIG_DIR, 'event_pos_num_memes_z.png'))
    plot_event(neg_z, 'num_memes_z', 'Event: Negative news vs normalized (z) meme volume', os.path.join(FIG_DIR, 'event_neg_num_memes_z.png'))

    # Rolling baseline ratio event study
    pos_r, neg_r = event_study_value(panel, pos_feature='mean_pos', neg_feature='mean_neg', value_col='num_memes_rel', window=3)
    plot_event(pos_r, 'num_memes_rel', 'Event: Positive news vs relative meme volume', os.path.join(FIG_DIR, 'event_pos_num_memes_rel.png'))
    plot_event(neg_r, 'num_memes_rel', 'Event: Negative news vs relative meme volume', os.path.join(FIG_DIR, 'event_neg_num_memes_rel.png'))

    # Quick console summary
    def _summ(df: pd.DataFrame, col: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        return df.groupby('tau')[col].agg(['mean', 'median', 'count'])

    print("\nPositive events (z):\n", _summ(pos_z, 'num_memes_z').to_string())
    print("\nNegative events (z):\n", _summ(neg_z, 'num_memes_z').to_string())
    print("\nPositive events (rel):\n", _summ(pos_r, 'num_memes_rel').to_string())
    print("\nNegative events (rel):\n", _summ(neg_r, 'num_memes_rel').to_string())

if __name__ == '__main__':
    run()
