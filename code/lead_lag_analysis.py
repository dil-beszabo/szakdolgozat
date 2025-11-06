import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = "/Users/beszabo/bene/topicality-online"
DERIVED_DIR = os.path.join(REPO_ROOT, "data", "derived")
FIG_DIR = os.path.join(REPO_ROOT, "figures")
PANEL_CSV = os.path.join(DERIVED_DIR, "company_weekly_panel.csv")

os.makedirs(FIG_DIR, exist_ok=True)

# ---------------- Correlations across lags ---------------- #

def xcorr_by_company(panel: pd.DataFrame, feature: str, max_lag: int = 4) -> pd.DataFrame:
    rows = []
    for company, g in panel.groupby('company'):
        g = g.sort_values('week_start')
        y = g['num_memes_z'].astype(float).values
        for k in range(1, max_lag + 1):
            x_lag = g[f'{feature}_L{k}'].values
            mask = ~np.isnan(x_lag)
            if mask.sum() > 2:
                r = np.corrcoef(x_lag[mask], y[mask])[0,1]
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

def event_study(panel: pd.DataFrame, pos_feature: str, neg_feature: str, window: int = 3, monthly: bool = False):
    rows_pos, rows_neg = [], []
    for company, g in panel.groupby('company'):
        if monthly:
            g = g.sort_values('month_start').reset_index(drop=True)
        else:
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

# ---------------- Main ---------------- #

def run():
    panel = pd.read_csv(PANEL_CSV, parse_dates=['week_start'])
    # Cross-correlations
    for feat in ['sentiment_score', 'mean_pos', 'mean_neg', 'non_neutral_share', 'num_articles']:
        df = xcorr_by_company(panel, feat, max_lag=4)
        out = os.path.join(FIG_DIR, f'xcorr_{feat}.png')
        plot_xcorr(df, f'Lead-Lag: {feat} vs meme_spike', out)
        print(f'Saved {out}')
    # Event study
    df_pos, df_neg = event_study(panel, pos_feature='mean_pos', neg_feature='mean_neg', window=3)
    plot_event(df_pos, 'meme_spike', 'Event: Positive news vs meme_spike', os.path.join(FIG_DIR, 'event_pos_meme_spike.png'))
    plot_event(df_neg, 'meme_spike', 'Event: Negative news vs meme_spike', os.path.join(FIG_DIR, 'event_neg_meme_spike.png'))
    plot_event(df_pos, 'num_memes', 'Event: Positive news vs num_memes', os.path.join(FIG_DIR, 'event_pos_num_memes.png'))
    plot_event(df_neg, 'num_memes', 'Event: Negative news vs num_memes', os.path.join(FIG_DIR, 'event_neg_num_memes.png'))
    print('Event-study plots saved.')

if __name__ == '__main__':
    run()
