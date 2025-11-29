import pandas as pd

def check_required_columns(panel: pd.DataFrame,
                           required: list[str],
                           optional: list[str] | None = None) -> None:
    """Assert that all *required* columns are present.
    Optional columns are ignored but reported if missing."""
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    if optional:
        miss_opt = [c for c in optional if c not in panel.columns]
        if miss_opt:
            print(f"[warn] optional columns missing: {miss_opt}")

def summarize_missingness(panel: pd.DataFrame,
                          cols: list[str]) -> pd.DataFrame:
    """Return a tiny frame with non-null share for each column."""
    return pd.Series({c: panel[c].notna().mean() for c in cols},
                     name="non_null_share").to_frame()

def continuity_by_year(panel: pd.DataFrame,
                       var: str) -> pd.DataFrame:
    """Share of weeks where *var* >0 by (company, ISO-year)."""
    tmp = panel.copy()
    tmp["year"] = tmp["week_start"].dt.year
    return (tmp.groupby(["company", "year"])[var]
               .apply(lambda s: (s.fillna(0) > 0).mean())
               .rename("share_weeks_with_val")
               .reset_index())