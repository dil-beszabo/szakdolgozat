"""Build weekly, brand-level panel data by merging NYT article sentiment and meme activity.

This module:
- parses NYT article text files, scores per-article sentiment, aggregates to weekly per-brand
- scores meme images with CLIP + OCR→FinBERT, aggregates to weekly per-brand
- joins the two sides, balances the panel to brand-week grid, adds transforms and lags
"""
import os
import re
from typing import List, Dict, Tuple, Optional
import multiprocessing

# --------------- Third-party deps ---------------- #
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
import cv2
from transformers import CLIPProcessor, CLIPModel
import easyocr

import numpy as np
import pandas as pd

# --------------- Local imports ------------------- #
from process_nyt_articles import (
    _simple_key,
    get_finbert,
    score_article_company_fair,
    META_HEADERS,
    preprocess_article_text,
    split_articles,
)
from lead_lag_analysis import add_normalizations

# -------------------------- Config -------------------------- #
REPO_ROOT = "/Users/beszabo/bene/szakdolgozat"
NYT_DIR = os.path.join(REPO_ROOT, "data", "nyt")
PRED_IMG_DIR = os.path.join(REPO_ROOT, "data", "prediction_images")
DERIVED_DIR = os.path.join(REPO_ROOT, "data", "derived")

NYT_OUT = os.path.join(DERIVED_DIR, "nyt_weekly_sentiment.csv")
MEMES_OUT = os.path.join(DERIVED_DIR, "memes_weekly_activity.csv")
PANEL_OUT = os.path.join(DERIVED_DIR, "company_weekly_panel_enriched.csv")
ANALYSIS_OUT = os.path.join(DERIVED_DIR, "company_weekly_panel_analysis_ready.csv")

# Minimum publication date to include NYT articles in aggregates
NYT_MIN_DATE = pd.Timestamp("2023-01-01")

# Per-image sentiment cache
CACHE_CSV = os.path.join(DERIVED_DIR, "meme_image_sentiment_cache.csv")

# Ensure derived directory
os.makedirs(DERIVED_DIR, exist_ok=True)

# ------------------- NYT parsing and scoring ---------------- #

_NYT_WEEKLY_EMPTY_COLUMNS = [
    "company",
    "week_start",
    "mean_pos",
    "mean_neu",
    "mean_neg",
    "NYT_mention",
    "sentiment_score",
    "non_neutral_share",
]


def _empty_nyt_weekly_df() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_NYT_WEEKLY_EMPTY_COLUMNS))


def _parse_pub_date(lines: List[str]) -> Optional[pd.Timestamp]:
    """Parse publication date from 'Publication date:' line with a few known formats."""
    date_line = next((l for l in lines if l.startswith("Publication date:")), None)
    if not date_line:
        return None
    date_str = date_line.replace("Publication date:", "").strip()
    for fmt in (None, "%Y-%m-%d", "%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y"):
        try:
            if fmt is None:
                return pd.to_datetime(date_str, errors="raise")
            return pd.to_datetime(pd.to_datetime(date_str, format=fmt).date())
        except Exception:
            continue
    return None


def _extract_title_and_body(lines: List[str]) -> Tuple[str, str]:
    """Extract article title and body from a split block."""
    title_line = next((l for l in lines if l.startswith("Title:")), None)
    title = title_line.replace("Title:", "").strip() if title_line else (lines[0] if lines else "")

    ft_idx = next((i for i, l in enumerate(lines) if l.startswith("Full text:")), None)
    body = ""
    if ft_idx is not None:
        text_lines: List[str] = []
        for l in lines[ft_idx:]:
            if any(l.startswith(h) for h in META_HEADERS) and not l.startswith("Full text:"):
                break
            if l.startswith("Full text:"):
                l = l[len("Full text:"):].strip()
            text_lines.append(l)
        body = " ".join(text_lines).strip()
    return title, body


def _score_text_for_company(text: str, company_key: str) -> Dict[str, float]:
    """Preprocess and score text with company-aware FinBERT routing."""
    pre = preprocess_article_text(text)
    probs = score_article_company_fair(pre, company_key)
    return {
        "pos": probs.get("Positive", 0.0),
        "neu": probs.get("Neutral", 0.0),
        "neg": probs.get("Negative", 0.0),
    }


def _process_nyt_file_for_sentiment(filepath: str) -> List[Dict[str, float]]:
    """Worker: parse a single NYT file into per-article sentiment rows for that company."""
    # Init FinBERT per worker process
    clf = get_finbert()

    # Brand key from filename; keep inner hyphens, drop trailing -<digits>
    base = os.path.splitext(os.path.basename(filepath))[0]
    base = re.sub(r"-\d+$", "", base)
    company_key = _simple_key(base)
    company_article_probs = []

    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()
    
    for block in split_articles(raw):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        title, body = _extract_title_and_body(lines)
        pub_date = _parse_pub_date(lines)
        text = f"{title} {body}".strip()
        if not text or pub_date is None:
            continue

        probs = _score_text_for_company(text, company_key)
        company_article_probs.append({
            "company": company_key,
            "date": pd.to_datetime(pub_date).normalize(),
            "pos": probs["pos"],
            "neu": probs["neu"],
            "neg": probs["neg"],
        })

    return company_article_probs


def build_nyt_weekly(nyt_dir: str, num_processes: int = None) -> pd.DataFrame:
    """Parallel-parse NYT files and aggregate to weekly per company."""
    nyt_files = sorted(
        os.path.join(nyt_dir, f) for f in os.listdir(nyt_dir) if f.endswith(".txt")
    )
    
    if num_processes is None:
        num_processes = os.cpu_count() or 1
    print(f"Processing {len(nyt_files)} NYT files in parallel using {num_processes} processes...")

    with multiprocessing.Pool(processes=num_processes) as pool:
        results_lists = pool.map(_process_nyt_file_for_sentiment, nyt_files)
    
    rows = [row for lst in results_lists for row in lst]

    if not rows:
        return _empty_nyt_weekly_df()
    df = pd.DataFrame(rows)

    df = df[df["date"] >= NYT_MIN_DATE]
    if df.empty:
        return _empty_nyt_weekly_df()

    df["week_start"] = (df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="D"))
    grp = df.groupby(["company", "week_start"], as_index=False).agg(
        mean_pos=("pos", "mean"),
        mean_neu=("neu", "mean"),
        mean_neg=("neg", "mean"),
        NYT_mention=("date", "count"),
    )
    grp["sentiment_score"] = grp["mean_pos"] - grp["mean_neg"]
    grp["non_neutral_share"] = 1.0 - grp["mean_neu"]
    return grp

# ----------------- Memes weekly activity builder ------------- #

def _find_timestamp_column(df: pd.DataFrame) -> str:
    candidates = ["created_utc", "created_at", "created", "timestamp", "date"]
    col = next((c for c in candidates if c in df.columns), None)
    if col:
        return col
    raise ValueError("No timestamp column found in predictions CSV.")


def _find_path_or_company_columns(df: pd.DataFrame) -> Tuple[str, str]:
    path_cols = ["path", "filepath", "image_path", "image", "filename"]
    company_cols = ["company", "brand", "label", "folder"]
    path_col = next((c for c in path_cols if c in df.columns), None)
    comp_col = next((c for c in company_cols if c in df.columns), None)
    return path_col, comp_col


def _company_from_path(path: str) -> str:
    if not isinstance(path, str):
        return ""
    # Expect .../prediction_images/<Company>/... or similar
    parts = re.split(r"[\\/]+", path)
    # Find the segment right after 'prediction_images'
    try:
        idx = [p.lower() for p in parts].index("prediction_images")
        if idx + 1 < len(parts):
            return _simple_key(parts[idx + 1])
    except ValueError:
        pass
    # Fallback: take the first directory-looking segment
    for p in parts:
        if p and not p.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            return _simple_key(p)
    return ""

# ---------------- Image + OCR sentiment helpers ------------- #

_clip_model = None
_clip_proc = None
_ocr_reader = None

# Prefer GPU/MPS if available
_DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)


def _get_clip():
    global _clip_model, _clip_proc
    if _clip_model is None:
        _clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_DEVICE)
        _clip_model.eval()
    return _clip_model, _clip_proc


def _clip_sentiment(path: str) -> Tuple[float, float]:
    """Return (pos, neg) probs from zero-shot CLIP."""
    try:
        img = Image.open(path).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError):
        return 0.5, 0.5  # neutral default
    model, proc = _get_clip()
    prompts = ["a negative meme", "a positive meme"]
    inputs = proc(text=prompts, images=img, return_tensors="pt", padding=True).to(_DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image.squeeze()
    probs = F.softmax(logits, dim=0).tolist()
    return probs[1], probs[0]  # pos, neg


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader


def _ocr_finbert_sentiment(path: str) -> Tuple[float, float]:
    """OCR the image and classify the extracted text with FinBERT; returns (pos, neg)."""
    reader = _get_ocr_reader()
    try:
        img = cv2.imread(str(path))
        if img is None:
            return 0.5, 0.5
        ocr_res = reader.readtext(img)
        text = " ".join([t[1] for t in ocr_res])
    except Exception:
        return 0.5, 0.5
    if not text.strip():
        return 0.5, 0.5
    clf = get_finbert()
    outs = clf([text], truncation=True, max_length=512, top_k=None)[0]
    scores = {d["label"]: d["score"] for d in outs}
    return scores.get("Positive", 0.0), scores.get("Negative", 0.0)


def _meme_sentiment(row) -> float:
    path = os.path.join(PRED_IMG_DIR, row["saved_path"]) if "saved_path" in row else row.get("path", "")
    pos_c, neg_c = _clip_sentiment(path)
    pos_t, neg_t = _ocr_finbert_sentiment(path)
    return ((pos_c - neg_c) + (pos_t - neg_t)) / 2.0


def build_memes_weekly(pred_dir: str) -> pd.DataFrame:
    # Prefer predictions_metadata.csv; fallback to predictions_manifest.csv
    enriched_meta_path = os.path.join(pred_dir, "enriched_predictions_metadata.csv")
    meta_path = enriched_meta_path if os.path.exists(enriched_meta_path) else os.path.join(pred_dir, "predictions_metadata.csv")
    mani_path = os.path.join(pred_dir, "predictions_manifest.csv")
    csv_path = meta_path if os.path.exists(meta_path) else mani_path
    df = pd.read_csv(csv_path)

    # Compute meme sentiment per-image (clip+ocr averaged)
    print("Scoring meme image sentiment (may take a while on CPU)...")
    df["meme_sentiment"] = df.apply(_meme_sentiment, axis=1)
    ts_col = _find_timestamp_column(df)
    path_col, comp_col = _find_path_or_company_columns(df)

    # Timestamp parse
    ts = df[ts_col]
    if np.issubdtype(ts.dtype, np.number):
        dt = pd.to_datetime(ts, unit="s", utc=True).dt.tz_localize(None)
    else:
        dt = pd.to_datetime(ts, errors="coerce", utc=True).dt.tz_localize(None)
    df["date"] = dt.dt.normalize()
    # Company
    if comp_col:
        df["company"] = df[comp_col].map(_simple_key)
    elif path_col:
        df["company"] = df[path_col].map(_company_from_path)
    else:
        raise ValueError("No company or path column found in predictions CSV.")

    df = df[~df["company"].isna() & (df["company"] != "")]
    df["week_start"] = (df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="D"))

    # Aggregate weekly counts (and optional score if exists)
    # Try to pick an engagement-like column if present
    engagement_col = next((c for c in ["engagement", "score", "upvotes"] if c in df.columns), None)
    if engagement_col:
        agg = df.groupby(["company", "week_start"], as_index=False).agg(
            num_memes=("date", "count"),
            mean_meme_sentiment=("meme_sentiment", "mean"),
            meme_engagement=(engagement_col, "mean"),
        )
    else:
        agg = df.groupby(["company", "week_start"], as_index=False).agg(
            num_memes=("date", "count"),
            mean_meme_sentiment=("meme_sentiment", "mean"),
        )
    # Spike rule: weekly num_memes > mean + 2*std within company
    def _spike_flag(s: pd.Series) -> pd.Series:
        thresh = s.quantile(0.90)  
        return (s > thresh).astype(int)
    agg["meme_spike"] = agg.groupby("company")["num_memes"].transform(_spike_flag)
    return agg

# -------------------------- Panel join ----------------------- #

def make_lags(panel: pd.DataFrame, by: str, date_col: str, cols: List[str], max_lag: int = 4) -> pd.DataFrame:
    panel = panel.sort_values([by, date_col]).copy()
    for c in cols:
        for k in range(1, max_lag + 1):
            panel[f"{c}_L{k}"] = panel.groupby(by)[c].shift(k)
    return panel


if __name__ == "__main__":
    # Orchestrate the build and save of NYT sentiment, meme activity, and the combined panel.

    print("Building NYT weekly sentiment...")
    nyt_weekly = build_nyt_weekly(NYT_DIR, num_processes=multiprocessing.cpu_count())
    nyt_weekly.to_csv(NYT_OUT, index=False)
    print("Saved:", NYT_OUT)
    print(nyt_weekly.head().to_string(index=False))

    print("\nBuilding memes weekly activity...")
  
    print("Building memes weekly activity (first run, this can be slow)...")
    memes_weekly = build_memes_weekly(PRED_IMG_DIR)
    memes_weekly.to_csv(MEMES_OUT, index=False)
    print("Saved:", MEMES_OUT)
    print(memes_weekly.head().to_string(index=False))

    print("\nJoining into weekly panel and creating lags...")
    panel = pd.merge(nyt_weekly, memes_weekly, on=["company", "week_start"], how="outer")
    # counts → 0; means stay NaN if no memes
    for c in ["NYT_mention", "num_memes"]:
        if c in panel.columns:
            panel[c] = panel[c].fillna(0).astype(int)

    # Save the raw merged/enriched panel for reference (backwards compatibility)
    panel.to_csv(PANEL_OUT, index=False)
    print("Saved (enriched merge):", PANEL_OUT)

    # Balance to full brand-week grid for analysis-ready output
    companies = sorted(panel["company"].dropna().unique().tolist())
    wk_min = panel["week_start"].min()
    wk_max = panel["week_start"].max()
    weeks = pd.date_range(wk_min, wk_max, freq="W-MON")
    grid = pd.MultiIndex.from_product([companies, weeks], names=["company", "week_start"])
    panel2 = (
        panel.set_index(["company", "week_start"])
        .reindex(grid)
        .reset_index()
    )

    # Re-apply fill rules after balancing
    for c in ["NYT_mention", "num_memes"]:
        if c in panel2.columns:
            panel2[c] = panel2[c].fillna(0).astype(int)

    # Log transforms and ISO helpers
    if "num_memes" in panel2.columns:
        panel2["log1p_meme_volume"] = np.log1p(panel2["num_memes"])
    if "meme_engagement" in panel2.columns:
        panel2["log1p_meme_engagement"] = np.log1p(panel2["meme_engagement"].fillna(0))
    iso = panel2["week_start"].dt.isocalendar()
    panel2["iso_year"] = iso.year.astype(int)
    panel2["iso_week"] = iso.week.astype(int)

    # Create lags for NYT and meme-side variables
    lag_cols = ["sentiment_score", "mean_pos", "mean_neg", "non_neutral_share", "NYT_mention",
                "num_memes", "mean_meme_sentiment", "meme_engagement"]
    lag_cols = [c for c in lag_cols if c in panel2.columns]
    panel2 = make_lags(panel2, by="company", date_col="week_start", cols=lag_cols, max_lag=4)

    # Add z/relative normalizations on the balanced panel
    print("\nAdding meme normalizations (if any) on balanced panel..")
    panel2 = add_normalizations(panel2)

    # Rename NYT fields for clarity
    panel2 = panel2.rename(columns={
        "sentiment_score": "nyt_sentiment",
        "mean_pos": "nyt_pos_share",
        "mean_neg": "nyt_neg_share",
        "mean_neu": "nyt_neu_share",
        "non_neutral_share": "nyt_non_neutral_share",
    })
    # Rename lag columns accordingly
    lag_rename_map = {}
    for k in range(1, 5):
        for old, new in [
            (f"sentiment_score_L{k}", f"nyt_sentiment_L{k}"),
            (f"mean_pos_L{k}", f"nyt_pos_share_L{k}"),
            (f"mean_neg_L{k}", f"nyt_neg_share_L{k}"),
            (f"non_neutral_share_L{k}", f"nyt_non_neutral_share_L{k}"),
        ]:
            if old in panel2.columns:
                lag_rename_map[old] = new
    if lag_rename_map:
        panel2 = panel2.rename(columns=lag_rename_map)

    # Save analysis-ready panel
    panel2.to_csv(ANALYSIS_OUT, index=False)
    print("Saved (analysis-ready):", ANALYSIS_OUT)
    print(panel2.head().to_string(index=False))
    print("\nDone. You can now run TWFE estimation on the analysis-ready panel.")


