import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# ---- Column naming convention (single source of truth) ---- #
NYT_POS = "mean_pos"
NYT_NEG = "mean_neg"
NYT_SENT = "sentiment_score"
NYT_NON_NEUTRAL = "non_neutral_share"
NYT_COUNT = "NYT_mention"

# outcome columns
MEME_COUNT = "num_memes"
MEME_Z = "num_memes_z"
MEME_REL = "num_memes_rel"

def xcorr_by_company(panel: pd.DataFrame, feature: str, max_lag: int = 4) -> pd.DataFrame:
    rows = []
    for company, g in panel.groupby('company'):
        g = g.sort_values('week_start')
        y = g[MEME_Z].astype(float).values
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
    plt.ylabel(f'Correlation with {MEME_Z}')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def corrmx_nyt(panel: pd.DataFrame, plot: bool = True, out_path: str = None) -> pd.DataFrame:
    """
    Correlation matrix for NYT sentiment & intensity variables.
    """
    nyt_vars = [
        "NYT_mention",
        "mean_pos",
        "mean_neg",
        "mean_neu",
        "sentiment_score",
        "non_neutral_share",
    ]
    
    cols = [c for c in nyt_vars if c in panel.columns]
    corr = panel[cols].corr()

    if plot:
        plt.figure(figsize=(7, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("NYT Sentiment/Intensity Correlation Matrix")
        plt.tight_layout()
        plt.xticks(rotation=45)
        if out_path:
            plt.savefig(out_path)
            plt.close()
        else:
            plt.show()

    return corr

def corrmx_meme(panel: pd.DataFrame, plot: bool = True, out_path: str = None) -> pd.DataFrame:
    """
    Correlation matrix for meme-side activity, engagement, and sentiment variables.
    """
    meme_vars = [
        "num_memes",
        "num_memes_z",
        "num_memes_rel",
        "log1p_meme_volume",
        "meme_engagement",
        "log1p_meme_engagement",
        "mean_meme_sentiment",
    ]
    
    cols = [c for c in meme_vars if c in panel.columns]
    corr = panel[cols].corr()

    if plot:
        plt.figure(figsize=(7, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Meme Activity / Engagement / Sentiment Correlation Matrix")
        plt.tight_layout()
        plt.xticks(rotation=45) 
        if out_path:
            plt.savefig(out_path)
            plt.close()
        else:
            plt.show()

    return corr

def corrmx_cross(panel: pd.DataFrame, plot: bool = True, out_path: str = None) -> pd.DataFrame:
    """
    Cross-correlation matrix between NYT variables and meme outcomes (same-week).
    """
    cross_vars = [
        "NYT_mention",
        "sentiment_score",
        "mean_pos",
        "mean_neg",
        "num_memes_z",
        "num_memes_rel",
        "log1p_meme_volume",
        "log1p_meme_engagement",
        "mean_meme_sentiment",
    ]
    
    cols = [c for c in cross_vars if c in panel.columns]
    corr = panel[cols].corr()

    if plot:
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Cross-Domain NYT ↔ Meme Correlation Matrix")
        plt.tight_layout()
        plt.xticks(rotation=45) 
        if out_path:
            plt.savefig(out_path)
            plt.close()
        else:
            plt.show()

    return corr


# --- Correlation matrices with p-values (Pearson & Spearman) --- #

def _corr_with_p_generic(df: pd.DataFrame, cols, method: str = "pearson"):
    """Return correlation and p-value matrices using the chosen method."""

    r_mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
    p_mat = pd.DataFrame(index=cols, columns=cols, dtype=float)

    corr_func = pearsonr if method == "pearson" else spearmanr

    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if j < i:
                r_mat.loc[c1, c2] = r_mat.loc[c2, c1]
                p_mat.loc[c1, c2] = p_mat.loc[c2, c1]
            else:
                pair = df[[c1, c2]].dropna()
                if len(pair) < 2:
                    r, p = np.nan, np.nan
                elif c1 == c2:
                    r, p = 1.0, 0.0
                else:
                    r, p = corr_func(pair[c1].values, pair[c2].values)
                # coerce to scalar (spearmanr can return 2x2 when inputs identical)
                try:
                    r = float(np.asarray(r).item())
                except Exception:
                    r = np.nan
                try:
                    p = float(np.asarray(p).item())
                except Exception:
                    p = np.nan
                r_mat.loc[c1, c2] = r
                p_mat.loc[c1, c2] = p

    return r_mat.astype(float), p_mat.astype(float)


def corr_with_p_pearson(df: pd.DataFrame, cols):
    """Pearson correlation (& p) matrix for selected columns."""
    return _corr_with_p_generic(df, cols, method="pearson")


def corr_with_p_spearman(df: pd.DataFrame, cols):
    """Spearman rank-correlation (& p) matrix for selected columns."""
    return _corr_with_p_generic(df, cols, method="spearman")
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
                        rows_pos.append({'company': company, 'tau': tau, 'meme_spike': g.loc[j, 'meme_spike'], MEME_COUNT: g.loc[j, MEME_COUNT]})
            # negative event
            if g.loc[i, neg_feature] >= neg_thresh:
                for tau in range(-window, window+1):
                    j = i + tau
                    if 0 <= j < len(g):
                        rows_neg.append({'company': company, 'tau': tau, 'meme_spike': g.loc[j, 'meme_spike'], MEME_COUNT: g.loc[j, MEME_COUNT]})
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
    """Add per-company normalizations for meme counts.

    Expects the raw count column to be present as `MEME_COUNT` ("num_memes").
    Adds:
        • `MEME_Z`  : z-score within company.
        • `MEME_REL`: ratio to the rolling mean of the previous 8 weeks.
    """
    if MEME_COUNT not in panel.columns:
        raise KeyError(f"Column '{MEME_COUNT}' is missing from the panel")

    panel = panel.sort_values(["company", "week_start"]).copy()

    # --- Handle truly missing meme weeks --------------------- #
    if "mean_meme_sentiment" in panel.columns:
        missing = (panel[MEME_COUNT] == 0) & (panel["mean_meme_sentiment"].isna())
    else:
        missing = panel[MEME_COUNT] == 0

    panel["_num_memes_obs"] = panel[MEME_COUNT].astype(float)
    panel.loc[missing, "_num_memes_obs"] = np.nan  # treat unobserved weeks as NaN

    stats = (
        panel.dropna(subset=["_num_memes_obs"])
        .groupby("company")["_num_memes_obs"]
        .agg(mu="mean", sd="std")
        .reset_index()
    )
    panel = panel.merge(stats, on="company", how="left")
    panel["sd"] = panel["sd"].replace(0, np.nan)

    panel[MEME_Z] = (panel["_num_memes_obs"] - panel["mu"]) / panel["sd"]
    panel[MEME_Z] = panel[MEME_Z].fillna(0.0)

    roll = (
        panel.groupby("company")[MEME_COUNT]
            .transform(lambda s: s.shift(1).rolling(8, min_periods=3).mean())
    )
    panel[MEME_REL] = panel[MEME_COUNT] / (roll.replace(0, np.nan))
    panel[MEME_REL] = panel[MEME_REL].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    return panel.drop(columns=["_num_memes_obs", "mu", "sd"], errors="ignore")


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

# ---------------- Notebook helper utilities (moved from notebook) ---------------- #

def plot_event_ci(df: pd.DataFrame, value_col: str, title: str, out_path: str):
    if df is None or df.empty:
        return
    agg = df.groupby("tau")[value_col].agg(["mean", "std", "count"]).reset_index()
    agg["se"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    agg["lo"] = agg["mean"] - 1.96 * agg["se"]
    agg["hi"] = agg["mean"] + 1.96 * agg["se"]

    plt.figure(figsize=(6, 4))
    plt.plot(agg["tau"], agg["mean"], marker="o", label="Mean")
    plt.fill_between(agg["tau"], agg["lo"], agg["hi"], alpha=0.2, label="95% CI")
    plt.axhline(0, color="gray", linewidth=1)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Weeks around event")
    plt.ylabel(f"Mean {value_col}")
    plt.title(title)
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_brand_timeseries(panel: pd.DataFrame,
                          company: str,
                          left_col: str = NYT_COUNT,
                          right_col: str = MEME_Z,
                          smooth: int = 0,
                          out_path: str | None = None):
    g = panel.loc[panel["company"] == company].sort_values("week_start").copy()
    if g.empty:
        return

    x = g["week_start"]
    left = g[left_col].astype(float)
    right = g[right_col].astype(float)

    if smooth and smooth > 1:
        left = left.rolling(window=smooth, min_periods=1).mean()
        right = right.rolling(window=smooth, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=(9, 3.5))
    color_left, color_right = "#1f77b4", "#d62728"

    ax1.plot(x, left, color=color_left, label=left_col)
    ax1.set_xlabel("Week")
    ax1.set_ylabel(left_col, color=color_left)
    ax1.tick_params(axis="y", labelcolor=color_left)

    ax2 = ax1.twinx()
    ax2.plot(x, right, color=color_right, label=right_col, alpha=0.85)
    ax2.set_ylabel(right_col, color=color_right)
    ax2.tick_params(axis="y", labelcolor=color_right)

    fig.suptitle(f"{company}: {left_col} vs {right_col}")
    fig.tight_layout()

    if out_path:
        plt.savefig(out_path)
        plt.close()
    else:
        plt.show()


def find_events(g: pd.DataFrame, feature: str, q: float) -> list[int]:
    """Return indices i in g (sorted by week_start) where feature >= q-quantile."""
    thresh = g[feature].quantile(q)
    return [i for i in range(len(g)) if pd.notna(g.loc[i, feature]) and g.loc[i, feature] >= thresh]


def enforce_non_overlap(event_idx: list[int], min_gap: int) -> list[int]:
    kept: list[int] = []
    last = -10_000
    for i in sorted(event_idx):
        if i - last > min_gap:
            kept.append(i)
            last = i
    return kept

# Compute deciles within company using rank to avoid ties issues
def _company_deciles(s: pd.Series, q: int = 10) -> pd.Series:
    r = s.rank(method='first')
    try:
        return pd.qcut(r, q, labels=False, duplicates='drop')
    except Exception:
        # Fallback to single bin if not enough unique values
        return pd.Series(0, index=s.index)

def event_study_from_indices(panel: pd.DataFrame, events: dict[str, list[int]], value_col: str, window: int) -> pd.DataFrame:
    rows = []
    for company, g in panel.groupby("company"):
        g = g.sort_values("week_start").reset_index(drop=True)
        idxs = events.get(company, [])
        n = len(g)
        for i in idxs:
            for tau in range(-window, window + 1):
                j = i + tau
                if 0 <= j < n and pd.notna(g.loc[j, value_col]):
                    rows.append({"company": company, "tau": tau, value_col: float(g.loc[j, value_col])})
    return pd.DataFrame(rows)


def build_event_dict(panel: pd.DataFrame, feature: str, q: float, non_overlapping: bool, shift: int = 0, min_gap: int = 3) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for company, g in panel.groupby("company"):
        g = g.sort_values("week_start").reset_index(drop=True)
        idxs = find_events(g, feature, q)
        if shift:
            idxs = [i + shift for i in idxs if 0 <= i + shift < len(g)]
        if non_overlapping:
            idxs = enforce_non_overlap(idxs, min_gap=min_gap)
        out[company] = idxs
    return out


def plot_diff_ci(event_df: pd.DataFrame, placebo_df: pd.DataFrame, value_col: str, title: str, out_path: str):
    if event_df is None or event_df.empty or placebo_df is None or placebo_df.empty:
        return
    e = event_df.groupby("tau")[value_col].agg(["mean", "std", "count"]).rename(columns={"mean": "e_mean", "std": "e_std", "count": "e_n"})
    p = placebo_df.groupby("tau")[value_col].agg(["mean", "std", "count"]).rename(columns={"mean": "p_mean", "std": "p_std", "count": "p_n"})
    agg = e.join(p, how="inner").reset_index()
    if agg.empty:
        return
    agg["e_se"] = agg["e_std"] / np.sqrt(agg["e_n"].clip(lower=1))
    agg["p_se"] = agg["p_std"] / np.sqrt(agg["p_n"].clip(lower=1))
    agg["diff"] = agg["e_mean"] - agg["p_mean"]
    agg["se_diff"] = np.sqrt(agg["e_se"] ** 2 + agg["p_se"] ** 2)
    agg["lo"] = agg["diff"] - 1.96 * agg["se_diff"]
    agg["hi"] = agg["diff"] + 1.96 * agg["se_diff"]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(agg["tau"], agg["diff"], marker="o", label="Event − Placebo")
    plt.fill_between(agg["tau"], agg["lo"], agg["hi"], alpha=0.2, label="95% CI")
    plt.axhline(0, color="gray", linewidth=1)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Weeks around event")
    plt.ylabel(f"Diff {value_col}")
    plt.title(title)
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    
# ---------- High-level analysis helpers ---------- #

def run_alt_outcome_event_studies(panel: pd.DataFrame, fig_dir: str) -> None:
    """
    Produce CI plots for num_memes_rel and log1p outcomes.
    Re-creates 'alternative outcome' section of the notebook.
    """
    # Ensure alt outcomes exist
    panel['log1p_meme_volume'] = np.log1p(panel[MEME_COUNT].fillna(0))
    if 'meme_engagement' in panel.columns:
        panel['log1p_meme_engagement'] = np.log1p(panel['meme_engagement'].fillna(0))
    
    # num_memes_rel events
    pos_rel, neg_rel = event_study_value(
        panel, pos_feature='mean_pos', neg_feature='mean_neg',
        value_col=MEME_REL, window=3
    )
    plot_event_ci(pos_rel, MEME_REL,
                  'Event: Positive news vs relative meme volume (95% CI)',
                  os.path.join(fig_dir, 'appendix/alt-outcomes/event_pos_num_memes_rel_ci.png'))
    plot_event_ci(neg_rel, MEME_REL,
                  'Event: Negative news vs relative meme volume (95% CI)',
                  os.path.join(fig_dir, 'appendix/alt-outcomes/event_neg_num_memes_rel_ci.png'))
    
    # log1p volume
    pos_logv, neg_logv = event_study_value(
        panel, pos_feature='mean_pos', neg_feature='mean_neg',
        value_col='log1p_meme_volume', window=3
    )
    plot_event_ci(pos_logv, 'log1p_meme_volume',
                  'Event: Positive news vs log1p meme volume (95% CI)',
                  os.path.join(fig_dir, 'appendix/alt-outcomes/event_pos_log1p_meme_volume_ci.png'))
    plot_event_ci(neg_logv, 'log1p_meme_volume',
                  'Event: Negative news vs log1p meme volume (95% CI)',
                  os.path.join(fig_dir, 'appendix/alt-outcomes/event_neg_log1p_meme_volume_ci.png'))
    
    # log1p engagement if available
    if 'log1p_meme_engagement' in panel.columns:
        pos_loge, neg_loge = event_study_value(
            panel, pos_feature='mean_pos', neg_feature='mean_neg',
            value_col='log1p_meme_engagement', window=3
        )
        plot_event_ci(pos_loge, 'log1p_meme_engagement',
                      'Event: Positive news vs log1p meme engagement (95% CI)',
                      os.path.join(fig_dir, 'appendix/alt-outcomes/event_pos_log1p_meme_engagement_ci.png'))
        plot_event_ci(neg_loge, 'log1p_meme_engagement',
                      'Event: Negative news vs log1p meme engagement (95% CI)',
                      os.path.join(fig_dir, 'appendix/alt-outcomes/event_neg_log1p_meme_engagement_ci.png'))
    
    print('Saved alternative outcome CI event-study plots.')

def run_event_alignment_diagnostics(panel: pd.DataFrame, fig_dir: str) -> None:
    """No-overlap, ±1-week shift, and week-demeaned diagnostics."""
    win = 3
    
    # 1) Non-overlapping event windows
    pos_events_noov = build_event_dict(panel, 'mean_pos', q=0.90, non_overlapping=True, min_gap=win)
    neg_events_noov = build_event_dict(panel, 'mean_neg', q=0.90, non_overlapping=True, min_gap=win)
    
    pos_noov = event_study_from_indices(panel, pos_events_noov, MEME_Z, window=win)
    neg_noov = event_study_from_indices(panel, neg_events_noov, MEME_Z, window=win)
    
    plot_event_ci(pos_noov, MEME_Z,
                  'Event (no-overlap): Positive news vs num_memes_z',
                  os.path.join(fig_dir, 'appendix/alignment-shifts/event_pos_num_memes_z_ci_nooverlap.png'))
    plot_event_ci(neg_noov, MEME_Z,
                  'Event (no-overlap): Negative news vs num_memes_z',
                  os.path.join(fig_dir, 'appendix/alignment-shifts/event_neg_num_memes_z_ci_nooverlap.png'))
    
    print(f'Events kept (pos/neg): {sum(len(v) for v in pos_events_noov.values())} {sum(len(v) for v in neg_events_noov.values())}')
    
    # 2) Alignment shift tests (shift events by -1 and +1 week)
    pos_events_m1 = build_event_dict(panel, 'mean_pos', q=0.90, non_overlapping=True, shift=-1, min_gap=win)
    pos_events_p1 = build_event_dict(panel, 'mean_pos', q=0.90, non_overlapping=True, shift=+1, min_gap=win)
    
    for lab, ev in [('shift_m1', pos_events_m1), ('shift_p1', pos_events_p1)]:
        dfv = event_study_from_indices(panel, ev, MEME_Z, window=win)
        out = os.path.join(fig_dir, f'appendix/alignment-shifts/event_pos_num_memes_z_ci_{lab}.png')
        plot_event_ci(dfv, MEME_Z, f'Event (pos, {lab}): num_memes_z', out)
        if not dfv.empty:
            tau0_mean = dfv[dfv['tau'] == 0][MEME_Z].mean()
            print(f'{lab} tau0 mean= {tau0_mean}')
    
    # 3) Week-demeaned outcomes
    wk_mean = panel.groupby('week_start')[MEME_Z].transform('mean')
    panel['num_memes_z_dm'] = panel[MEME_Z] - wk_mean
    
    pos_dm = event_study_from_indices(panel, pos_events_noov, 'num_memes_z_dm', window=win)
    neg_dm = event_study_from_indices(panel, neg_events_noov, 'num_memes_z_dm', window=win)
    
    plot_event_ci(pos_dm, 'num_memes_z_dm',
                  'Event (demeaned): Positive news vs num_memes_z',
                  os.path.join(fig_dir, 'appendix/week-demeaned/event_pos_num_memes_z_demeaned_ci.png'))
    plot_event_ci(neg_dm, 'num_memes_z_dm',
                  'Event (demeaned): Negative news vs num_memes_z',
                  os.path.join(fig_dir, 'appendix/week-demeaned/event_neg_num_memes_z_demeaned_ci.png'))
    
    print('Saved no-overlap, shift tests, and demeaned-week diagnostics.')

def run_nyt_spike_event_study(panel: pd.DataFrame, fig_dir: str) -> None:
    """Positive / negative NYT-tone spikes vs meme volume."""
    win = 3
    
    # Event definition: top 10% NYT_mention per company, non-overlapping
    mention_events = build_event_dict(panel, NYT_COUNT, q=0.90, non_overlapping=True, min_gap=win)
    
    # Unconditional on tone
    m_ev = event_study_from_indices(panel, mention_events, MEME_Z, window=win)
    plot_event_ci(m_ev, MEME_Z,
                  'Event (mentions spikes): num_memes_z',
                  os.path.join(fig_dir, 'appendix/mentions-as-events/event_mentions_num_memes_z_ci.png'))
    
    # Split by tone sign at event week (sentiment_score >= 0 vs < 0)
    pos_split: dict[str, list[int]] = {}
    neg_split: dict[str, list[int]] = {}
    
    for company, g in panel.groupby('company'):
        g = g.sort_values('week_start').reset_index(drop=True)
        idxs = mention_events.get(company, [])
        pos_idx = [i for i in idxs if pd.notna(g.loc[i, 'sentiment_score']) and g.loc[i, 'sentiment_score'] >= 0]
        neg_idx = [i for i in idxs if pd.notna(g.loc[i, 'sentiment_score']) and g.loc[i, 'sentiment_score'] < 0]
        pos_split[company] = pos_idx
        neg_split[company] = neg_idx
    
    m_pos = event_study_from_indices(panel, pos_split, MEME_Z, window=win)
    m_neg = event_study_from_indices(panel, neg_split, MEME_Z, window=win)
   
    plot_event_ci(m_pos, MEME_Z,
                  'Event (mentions spikes, pos tone): num_memes_z',
                  os.path.join(fig_dir, 'appendix/mentions-as-events/event_mentions_pos_num_memes_z_ci.png'))
    plot_event_ci(m_neg, MEME_Z,
                  'Event (mentions spikes, neg tone): num_memes_z',
                  os.path.join(fig_dir, 'appendix/mentions-as-events/event_mentions_neg_num_memes_z_ci.png'))
    
    print('Saved NYT mention spike event-study plots.')

# ---- Placebo helpers ---- #
def build_placebo_panel(panel: pd.DataFrame,
                        events: dict[str, list[int]],
                        value_col: str,
                        window: int = 3) -> pd.DataFrame:
    """Return placebo windows matched on NYT_mention decile."""
    rows = []
    for company, g in panel.groupby("company"):
        g = g.sort_values("week_start").reset_index(drop=True)
        event_idxs = events.get(company, [])
        if not event_idxs:
            continue
        
        # Compute deciles for NYT_mention within this company
        g['nyt_decile'] = _company_deciles(g[NYT_COUNT], q=10)
        
        # For each event, find placebo candidates in same decile
        n = len(g)
        for i_event in event_idxs:
            decile = g.loc[i_event, 'nyt_decile']
            event_nyt = g.loc[i_event, NYT_COUNT]
            
            # Candidate placebos: same decile, exclude event window
            candidates = [
                j for j in range(n)
                if g.loc[j, 'nyt_decile'] == decile
                and abs(j - i_event) > window
            ]
            
            # Pick closest by NYT_mention value
            if candidates:
                diffs = [abs(g.loc[j, NYT_COUNT] - event_nyt) for j in candidates]
                best = candidates[np.argmin(diffs)]
                
                # Extract placebo window
                for tau in range(-window, window + 1):
                    j = best + tau
                    if 0 <= j < n and pd.notna(g.loc[j, value_col]):
                        rows.append({
                            "company": company,
                            "tau": tau,
                            value_col: float(g.loc[j, value_col])
                        })
    
    return pd.DataFrame(rows)

rng = np.random.default_rng(42)

# Compute deciles within company using rank to avoid ties issues
def _company_deciles(s: pd.Series, q: int = 10) -> pd.Series:
    r = s.rank(method='first')
    try:
        return pd.qcut(r, q, labels=False, duplicates='drop')
    except Exception:
        # Fallback to single bin if not enough unique values
        return pd.Series(0, index=s.index)

def event_study_from_indices(panel: pd.DataFrame, events: dict[str, list[int]], value_col: str, window: int) -> pd.DataFrame:
    rows = []
    for company, g in panel.groupby("company"):
        g = g.sort_values("week_start").reset_index(drop=True)
        idxs = events.get(company, [])
        n = len(g)
        for i in idxs:
            for tau in range(-window, window + 1):
                j = i + tau
                if 0 <= j < n and pd.notna(g.loc[j, value_col]):
                    rows.append({"company": company, "tau": tau, value_col: float(g.loc[j, value_col])})
    return pd.DataFrame(rows)


def run_placebo_event_study(panel: pd.DataFrame,
                            pos_events: dict, neg_events: dict,
                            fig_dir: str, window: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plot placebo CIs and return the two DF objects."""
    # Build placebo windows matched on NYT_mention deciles
    pos_placebo = build_placebo_panel(panel, pos_events, value_col=MEME_Z, window=window)
    neg_placebo = build_placebo_panel(panel, neg_events, value_col=MEME_Z, window=window)

    # Plot and quick comparison at tau=0
    plot_event_ci(pos_placebo, MEME_Z,
                  'Placebo (pos events, matched by NYT decile)',
                  os.path.join(fig_dir, 'appendix/placebo/placebo_pos_num_memes_z_ci.png'))
    plot_event_ci(neg_placebo, MEME_Z,
                  'Placebo (neg events, matched by NYT decile)',
                  os.path.join(fig_dir, 'appendix/placebo/placebo_neg_num_memes_z_ci.png'))

    if not pos_placebo.empty:
        tau0_mean = float(pos_placebo[pos_placebo['tau'] == 0][MEME_Z].mean())
        print(f'Placebo pos tau0 mean = {tau0_mean}')
    if not neg_placebo.empty:
        tau0_mean = float(neg_placebo[neg_placebo['tau'] == 0][MEME_Z].mean())
        print(f'Placebo neg tau0 mean = {tau0_mean}')
    
    return pos_placebo, neg_placebo

def run_event_placebo_diff(event_df: pd.DataFrame,
                           placebo_df: pd.DataFrame,
                           value_col: str, title: str,
                           out_path: str) -> None:
    """Δ-in-CI plot."""
    plot_diff_ci(event_df, placebo_df, value_col, title, out_path)