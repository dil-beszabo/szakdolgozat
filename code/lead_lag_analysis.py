import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def xcorr_by_company(panel: pd.DataFrame, feature: str, max_lag: int = 4) -> pd.DataFrame:
    rows = []
    for company, g in panel.groupby('company'):
        g = g.sort_values('week_start')
        y = g['num_memes_z'].astype(float).values
        for k in range(1, max_lag + 1):
            col = f'{feature}_L{k}'
            if col not in g.columns:
                continue
            x_lag = g[col].astype(float).values
            # mask out NaNs in both series
            mask = (~np.isnan(x_lag)) & (~np.isnan(y))
            if mask.sum() > 2:
                x_obs = x_lag[mask]
                y_obs = y[mask]
                # skip if either series has zero variance (avoids divide-by-zero warnings)
                if np.nanstd(x_obs) == 0 or np.nanstd(y_obs) == 0:
                    continue
                r = float(np.corrcoef(x_obs, y_obs)[0, 1])
                rows.append({'company': company, 'lag': k, 'r': r})
    return pd.DataFrame(rows)


def plot_xcorr(df: pd.DataFrame, title: str, out_path: str):
    # summarize across companies by median and IQR
    summary = df.groupby('lag')['r'].agg(['median', lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
    summary.columns = ['median', 'q25', 'q75']
    plt.figure(figsize=(6,4))
    plt.plot(summary.index, summary['median'], marker='o')
    plt.fill_between(summary.index, summary['q25'], summary['q75'], alpha=0.2)
    plt.axhline(0, color='gray', linewidth=1)
    plt.xlabel('Lag (weeks)')
    plt.ylabel('Correlation with num_memes_z')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ---------------- Event study ---------------- #

def event_study(panel: pd.DataFrame, pos_feature: str, neg_feature: str, window: int = 3):
    rows_pos, rows_neg = [], []
    for company, g in panel.groupby('company'):
        g = g.sort_values('week_start').reset_index(drop=True)
        # thresholds per company
        pos_thresh = g[pos_feature].quantile(0.90)
        neg_thresh = g[neg_feature].quantile(0.90)
        for i in range(len(g)):
            # positive event
            if g.loc[i, pos_feature] >= pos_thresh:
                for tau in range(-window, window+1):
                    j = i + tau
                    if 0 <= j < len(g):
                        rows_pos.append({'company': company, 'tau': tau, 'meme_spike': g.loc[j, 'meme_spike'], 'num_memes': g.loc[j, 'num_memes']})
            # negative event
            if g.loc[i, neg_feature] >= neg_thresh:
                for tau in range(-window, window+1):
                    j = i + tau
                    if 0 <= j < len(g):
                        rows_neg.append({'company': company, 'tau': tau, 'meme_spike': g.loc[j, 'meme_spike'], 'num_memes': g.loc[j, 'num_memes']})
    df_pos = pd.DataFrame(rows_pos)
    df_neg = pd.DataFrame(rows_neg)
    return df_pos, df_neg


def plot_event(df: pd.DataFrame, value_col: str, title: str, out_path: str):
    if df.empty:
        return
    summary = df.groupby('tau')[value_col].agg(['mean', 'count'])
    plt.figure(figsize=(6,4))
    plt.plot(summary.index, summary['mean'], marker='o')
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Weeks around event')
    plt.ylabel(f'Mean {value_col}')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def add_normalizations(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["company", "week_start"]).copy()
    # z-score per company
    mu = panel.groupby("company")["num_memes"].transform("mean")
    sd = panel.groupby("company")["num_memes"].transform("std").replace(0, np.nan)
    panel["num_memes_z"] = (panel["num_memes"] - mu) / sd
    panel["num_memes_z"] = panel["num_memes_z"].fillna(0.0)
    # ratio to rolling baseline (previous 8 weeks)
    def _rolling_mean(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(window=8, min_periods=3).mean()

    roll = panel.groupby("company")["num_memes"].apply(_rolling_mean).reset_index(level=0, drop=True)
    panel["num_memes_rel"] = panel["num_memes"] / (roll.replace(0, np.nan))
    panel["num_memes_rel"] = panel["num_memes_rel"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return panel


def event_study_value(panel: pd.DataFrame, pos_feature: str, neg_feature: str, value_col: str, window: int = 3,
                      pos_q: float = 0.90, neg_q: float = 0.90):
    rows_pos, rows_neg = [], []
    for company, g in panel.groupby('company'):
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