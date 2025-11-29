import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Column naming convention (single source of truth) ---- #
NYT_POS = "nyt_pos_share"
NYT_NEG = "nyt_neg_share"
NYT_SENT = "nyt_sentiment"
NYT_NON_NEUTRAL = "nyt_non_neutral_share"
NYT_COUNT = "NYT_mention"  # keep original produced by build_weekly_panel

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
    Re-creates ‘alternative outcome’ section of the notebook.
    """
    # (→ paste the existing logic, but use MEME_REL etc. constants)

def run_event_alignment_diagnostics(panel: pd.DataFrame, fig_dir: str) -> None:
    """No-overlap, ±1-week shift, and week-demeaned diagnostics."""
    # (→ paste block 5)

def run_nyt_spike_event_study(panel: pd.DataFrame, fig_dir: str) -> None:
    """Positive / negative NYT-tone spikes vs meme volume."""
    # (→ your new NYT_POS / NYT_NEG block)

# ---- Placebo helpers ---- #
def build_placebo_panel(panel: pd.DataFrame,
                        events: dict[str, list[int]],
                        value_col: str,
                        window: int = 3) -> pd.DataFrame:
    """Return placebo windows matched on NYT_mention decile."""
    # (→ move placebo_from_events here)

def run_placebo_event_study(panel: pd.DataFrame,
                            pos_events: dict, neg_events: dict,
                            fig_dir: str, window: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plot placebo CIs and return the two DF objects."""
    # (→ wrapper around build_placebo_panel + plot_event_ci)

def run_event_placebo_diff(event_df: pd.DataFrame,
                           placebo_df: pd.DataFrame,
                           value_col: str, title: str,
                           out_path: str) -> None:
    """Δ-in-CI plot."""
    # (→ existing plot_diff_ci block)