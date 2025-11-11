import os
from pathlib import Path
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd


ROOT = Path("/Users/beszabo/bene/szakdolgozat")
DERIVED = ROOT / "data" / "derived"
PANEL_CSV = DERIVED / "company_weekly_panel_analysis_ready.csv"


def compute_overview(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)

    def count_ok(req: List[str]) -> Tuple[int, float]:
        ok = df[req].notna().all(axis=1).sum()
        return int(ok), float(ok / total if total else 0.0)

    req0 = ["NYT_mention", "nyt_sentiment"]
    req1 = req0 + [f"NYT_mention_L1", f"nyt_sentiment_L1"]
    req2 = req0 + [f"NYT_mention_L{k}" for k in (1, 2)] + [f"nyt_sentiment_L{k}" for k in (1, 2)]
    req3 = req0 + [f"NYT_mention_L{k}" for k in (1, 2, 3)] + [f"nyt_sentiment_L{k}" for k in (1, 2, 3)]
    req4 = req0 + [f"NYT_mention_L{k}" for k in (1, 2, 3, 4)] + [f"nyt_sentiment_L{k}" for k in (1, 2, 3, 4)]

    rows = []
    for name, req in [
        ("current_only", req0),
        ("current+L1", req1),
        ("current+L1..L2", req2),
        ("current+L1..L3", req3),
        ("current+L1..L4", req4),
    ]:
        n_ok, share_ok = count_ok(req)
        rows.append({"requirement": name, "rows_ok": n_ok, "share_ok": share_ok, "variables_required": ",".join(req)})

    return pd.DataFrame(rows)


def per_brand_coverage(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["company", "week_start"]).copy()
    # Flags
    df["has_sent"] = df["nyt_sentiment"].notna()
    df["has_mention"] = df["NYT_mention"].fillna(0) > 0

    # Coverage masks
    def mask_req(req: List[str]) -> pd.Series:
        return df[req].notna().all(axis=1)

    req1 = ["NYT_mention", "nyt_sentiment", "NYT_mention_L1", "nyt_sentiment_L1"]
    req2 = ["NYT_mention", "nyt_sentiment"] + [f"NYT_mention_L{k}" for k in (1, 2)] + [f"nyt_sentiment_L{k}" for k in (1, 2)]
    req3 = ["NYT_mention", "nyt_sentiment"] + [f"NYT_mention_L{k}" for k in (1, 2, 3)] + [f"nyt_sentiment_L{k}" for k in (1, 2, 3)]
    req4 = ["NYT_mention", "nyt_sentiment"] + [f"NYT_mention_L{k}" for k in (1, 2, 3, 4)] + [f"nyt_sentiment_L{k}" for k in (1, 2, 3, 4)]

    df["ok_L1"] = mask_req(req1)
    df["ok_L2"] = mask_req(req2)
    df["ok_L3"] = mask_req(req3)
    df["ok_L4"] = mask_req(req4)

    agg = (
        df.groupby("company")
        .agg(
            weeks_total=("week_start", "count"),
            weeks_with_sentiment=("has_sent", "sum"),
            weeks_with_mention_gt0=("has_mention", "sum"),
            rows_ok_L1=("ok_L1", "sum"),
            rows_ok_L2=("ok_L2", "sum"),
            rows_ok_L3=("ok_L3", "sum"),
            rows_ok_L4=("ok_L4", "sum"),
        )
        .reset_index()
    )

    for c in ["weeks_with_sentiment", "weeks_with_mention_gt0", "rows_ok_L1", "rows_ok_L2", "rows_ok_L3", "rows_ok_L4"]:
        agg[f"{c}_share"] = np.where(agg["weeks_total"] > 0, agg[c] / agg["weeks_total"], np.nan)

    return agg.sort_values("rows_ok_L4", ascending=False)


def sentiment_run_lengths(df: pd.DataFrame, min_run: int = 1) -> pd.DataFrame:
    # Count consecutive runs of weeks where nyt_sentiment is present
    rows = []
    for company, g in df.sort_values(["company", "week_start"]).groupby("company"):
        has = g["nyt_sentiment"].notna().astype(int)
        # Identify run groups
        grp_id = (has != has.shift()).cumsum()
        gg = pd.DataFrame({"has": has, "grp": grp_id})
        runs = gg.groupby(["grp", "has"]).size().reset_index(name="len")
        # Keep only runs where has == 1
        for _, r in runs[runs["has"] == 1].iterrows():
            if r["len"] >= min_run:
                rows.append({"company": company, "run_len": int(r["len"])})

    if not rows:
        return pd.DataFrame(columns=["company", "run_len", "num_runs"])

    out = pd.DataFrame(rows).groupby(["company", "run_len"]).size().reset_index(name="num_runs")
    return out.sort_values(["company", "run_len"], ascending=[True, False])


def missing_reason_breakdown(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Focus on the strictest spec: current + L1..L4 for both predictors
    req = ["NYT_mention", "nyt_sentiment"] + [f"NYT_mention_L{k}" for k in (1, 2, 3, 4)] + [f"nyt_sentiment_L{k}" for k in (1, 2, 3, 4)]
    mask = df[req].notna().all(axis=1)

    # For rows that FAIL, count missing-by-variable and top missing-combos
    fail = df.loc[~mask, req].copy()
    var_missing = Counter()
    combo_missing = Counter()

    for _, row in fail.iterrows():
        miss_vars = [c for c in req if pd.isna(row[c])]
        for v in miss_vars:
            var_missing[v] += 1
        combo_missing[tuple(sorted(miss_vars))] += 1

    var_df = pd.DataFrame([{"variable": v, "missing_rows": n} for v, n in var_missing.items()]).sort_values("missing_rows", ascending=False)
    combo_rows = [{"missing_combo": "|".join(list(k)), "missing_rows": n} for k, n in combo_missing.items()]
    combo_df = pd.DataFrame(combo_rows).sort_values("missing_rows", ascending=False)

    total = len(df)
    if total > 0 and not var_df.empty:
        var_df["missing_share"] = var_df["missing_rows"] / total
    if total > 0 and not combo_df.empty:
        combo_df["missing_share"] = combo_df["missing_rows"] / total

    return var_df, combo_df


def main():
    if not PANEL_CSV.exists():
        raise FileNotFoundError(f"Panel CSV not found at {PANEL_CSV}")

    df = pd.read_csv(PANEL_CSV, parse_dates=["week_start"])
    df = df.sort_values(["company", "week_start"]).reset_index(drop=True)

    # Basic context
    total_rows = len(df)
    n_companies = df["company"].nunique()
    date_min = pd.to_datetime(df["week_start"]).min()
    date_max = pd.to_datetime(df["week_start"]).max()

    print(f"Total rows (brand×week): {total_rows}")
    print(f"Companies: {n_companies}")
    print(f"Week range: {date_min.date()} → {date_max.date()}")

    # Overview coverage
    overview = compute_overview(df)
    print("\nCoverage overview:\n", overview.to_string(index=False))

    # Per-brand coverage
    per_brand = per_brand_coverage(df)

    # Sentiment run-lengths (consecutive weeks with sentiment available)
    runs = sentiment_run_lengths(df, min_run=1)

    # Missing reason breakdown for strict spec (current+L1..L4)
    var_missing_df, combo_missing_df = missing_reason_breakdown(df)

    # Per-brand mention rate (how often NYT_mention > 0)
    mention_rate = (
        df.assign(has_mention=(df["NYT_mention"].fillna(0) > 0))
        .groupby("company")["has_mention"]
        .mean()
        .rename("mention_week_share")
        .reset_index()
        .sort_values("mention_week_share", ascending=False)
    )

    # Save reports
    DERIVED.mkdir(parents=True, exist_ok=True)
    overview.to_csv(DERIVED / "sparsity_overview.csv", index=False)
    per_brand.to_csv(DERIVED / "sparsity_per_brand.csv", index=False)
    runs.to_csv(DERIVED / "sparsity_sentiment_run_lengths.csv", index=False)
    var_missing_df.to_csv(DERIVED / "sparsity_missing_variables.csv", index=False)
    combo_missing_df.to_csv(DERIVED / "sparsity_missing_combinations_top.csv", index=False)
    mention_rate.to_csv(DERIVED / "nyt_mention_week_share_by_brand.csv", index=False)

    # Console highlights
    print("\nSaved:")
    print(" - sparsity_overview.csv")
    print(" - sparsity_per_brand.csv")
    print(" - sparsity_sentiment_run_lengths.csv")
    print(" - sparsity_missing_variables.csv")
    print(" - sparsity_missing_combinations_top.csv")
    print(" - nyt_mention_week_share_by_brand.csv")

    if not per_brand.empty:
        median_mention_share = per_brand["weeks_with_mention_gt0_share"].median()
        median_sent_share = per_brand["weeks_with_sentiment_share"].median()
        print(f"\nMedian brand share of weeks with NYT mention > 0: {median_mention_share:.3f}")
        print(f"Median brand share of weeks with NYT sentiment present: {median_sent_share:.3f}")

        median_rows_ok_L1 = per_brand["rows_ok_L1_share"].median()
        median_rows_ok_L4 = per_brand["rows_ok_L4_share"].median()
        print(f"Median brand share of weeks meeting current+L1: {median_rows_ok_L1:.3f}")
        print(f"Median brand share of weeks meeting current+L1..L4: {median_rows_ok_L4:.3f}")


if __name__ == "__main__":
    main()


