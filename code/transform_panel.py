import os
import re
import numpy as np
import pandas as pd

ROOT = "/Users/beszabo/bene/szakdolgozat"
DERIVED = f"{ROOT}/data/derived"
PRED_DIR = f"{ROOT}/data/prediction_images"

panel = pd.read_csv(f"{DERIVED}/company_weekly_panel_enriched.csv", parse_dates=["week_start"])

# 1) Bring in weekly meme engagement
meta_path = f"{PRED_DIR}/enriched_predictions_metadata.csv"
fallback_path = f"{PRED_DIR}/predictions_metadata.csv"
src = meta_path if os.path.exists(meta_path) else fallback_path
meta = pd.read_csv(src)

# timestamp → date → week_start
ts_col = next(c for c in ["created_utc","created_at","created","timestamp","date"] if c in meta.columns)
ts = meta[ts_col]
dt = pd.to_datetime(ts, unit="s", utc=True, errors="ignore") if np.issubdtype(ts.dtype, np.number) else pd.to_datetime(ts, utc=True, errors="coerce")
meta["date"] = dt.dt.tz_localize(None).dt.normalize()
meta["week_start"] = meta["date"] - pd.to_timedelta(meta["date"].dt.dayofweek, unit="D")

# company: use column if present, else derive from image path
if "company" not in meta.columns:
    path_col = next((c for c in ["path","filepath","image_path","image","filename","saved_path"] if c in meta.columns), None)
    def _simple_key(s): 
        s = re.sub(r"[^a-z0-9]+", "", str(s).lower()); 
        return s
    def _company_from_path(p):
        parts = re.split(r"[\\/]+", str(p))
        try:
            idx = [q.lower() for q in parts].index("prediction_images")
            if idx + 1 < len(parts):
                return _simple_key(parts[idx + 1])
        except ValueError:
            pass
        return _simple_key(parts[0]) if parts else ""
    meta["company"] = meta[path_col].map(_company_from_path)

# choose engagement column (score preferred)
eng_col = next(c for c in ["engagement","score","upvotes"] if c in meta.columns)
eng_weekly = (meta
    .groupby(["company","week_start"], as_index=False)
    .agg(meme_engagement=(eng_col, "mean"))
)

# 2) Merge and BALANCE the panel (outer + full brand-week grid)
panel2 = panel.merge(eng_weekly, on=["company","week_start"], how="outer")

companies = sorted(panel2["company"].dropna().unique().tolist())
wk_min = panel2["week_start"].min()
wk_max = panel2["week_start"].max()
weeks = pd.date_range(wk_min, wk_max, freq="W-MON")
grid = pd.MultiIndex.from_product([companies, weeks], names=["company","week_start"])
panel2 = (panel2
    .set_index(["company","week_start"])
    .reindex(grid)
    .reset_index()
)

# 3) Fill rules: counts→0, means stay NaN
for c in ["num_memes","NYT_mention"]:
    if c in panel2.columns:
        panel2[c] = panel2[c].fillna(0).astype(int)

# 4) Transforms and helpers
if "num_memes" in panel2.columns:
    panel2["log1p_meme_volume"] = np.log1p(panel2["num_memes"])
if "meme_engagement" in panel2.columns:
    panel2["log1p_meme_engagement"] = np.log1p(panel2["meme_engagement"].fillna(0))

iso = panel2["week_start"].dt.isocalendar()
panel2["iso_year"] = iso.year.astype(int)
panel2["iso_week"] = iso.week.astype(int)

# 5) Lags for meme and NYT variables
def add_lags(df, group="company", date_col="week_start", cols=(), max_lag=4):
    df = df.sort_values([group, date_col]).copy()
    for c in cols:
        for k in range(1, max_lag+1):
            df[f"{c}_L{k}"] = df.groupby(group)[c].shift(k)
    return df

lag_cols = [c for c in [
    "NYT_mention","sentiment_score",
    "num_memes","mean_meme_sentiment","meme_engagement"
] if c in panel2.columns]
panel2 = add_lags(panel2, cols=lag_cols, max_lag=4)

# Optional renames for clarity
panel2 = panel2.rename(columns={
    "sentiment_score": "nyt_sentiment",
    "mean_pos": "nyt_pos_share",
    "mean_neg": "nyt_neg_share",
    "mean_neu": "nyt_neu_share",
    "non_neutral_share": "nyt_non_neutral_share",
})

# Rename lag columns to match the new base names
lag_rename_map = {}
for k in range(1, 5):
    old_new_pairs = [
        (f"sentiment_score_L{k}", f"nyt_sentiment_L{k}"),
        (f"mean_pos_L{k}", f"nyt_pos_share_L{k}"),
        (f"mean_neg_L{k}", f"nyt_neg_share_L{k}"),
        (f"non_neutral_share_L{k}", f"nyt_non_neutral_share_L{k}"),
    ]
    for old, new in old_new_pairs:
        if old in panel2.columns:
            lag_rename_map[old] = new
if lag_rename_map:
    panel2 = panel2.rename(columns=lag_rename_map)

panel2.to_csv(f"{DERIVED}/company_weekly_panel_analysis_ready.csv", index=False)
print("Saved company_weekly_panel_analysis_ready.csv")