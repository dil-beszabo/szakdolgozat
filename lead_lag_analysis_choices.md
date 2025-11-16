# Lead–Lag Analysis: Methods and Key Choices

This note summarizes the methodological choices implemented in `code/lead_lag_analysis.ipynb` to generate the figures and interpret results.

## Data and preprocessing

- Source panel: `data/panels/company_weekly_panel_analysis_ready.csv` (balanced, calendar-complete weekly panel).
- Time key: `week_start` (weekly frequency); grouping key: `company`.

## Outcome construction

- Main outcome: per‑company z‑score of meme volume with NaN-aware handling of true non-observation:
  - If `num_memes == 0` and (optionally) `mean_meme_sentiment` is missing, treat as unobserved (NaN) when averaging.
  - Compute per‑company mean and sd on observed weeks only; z‑score into `num_memes_z_es`.
  - Alias for plotting/legacy helpers: `num_memes_z = num_memes_z_es`.
- Alternative outcomes used in robustness:
  - `num_memes_rel` (relative normalization from panel utilities).
  - `log1p_meme_volume = log1p(num_memes)`.
  - `log1p_meme_engagement = log1p(meme_engagement)` if engagement is present.

## Event definition and windowing

- Positive events: company‑week at or above the 90th percentile of `mean_pos` within the company.
- Negative events: company‑week at or above the 90th percentile of `mean_neg` within the company.
- Event window: symmetric `w = 3` weeks, i.e., τ ∈ {−3,…,0,…,+3}.
- Unless stated otherwise, overlapping events are allowed; for diagnostics we enforce non‑overlap with a minimum gap of `w`.

## Estimation and uncertainty
- For each event, collect the outcome at each τ, then average across events (equal weight per event).
- Uncertainty bands: pointwise 95% CIs using `mean ± 1.96 * (sd / sqrt(n))` across events at each τ.
- Note: CIs treat event windows as independent and do not adjust for serial correlation or multiple testing (kept simple by design).

## Diagnostics and robustness checks
1) Non‑overlap constraint
- Rebuild events with a minimum distance of `w` weeks and re‑estimate the event studies.
2) Alignment shift tests
- Shift identified positive events by −1 and +1 week; re‑estimate to verify alignment (τ=0 response should attenuate when misaligned).
3) Week‑demeaned outcome
- Demean `num_memes_z` by the cross‑sectional weekly mean to absorb common weekly shocks and re‑estimate.
4) Mentions‑spike events and tone split
- Define events as top 10% of `NYT_mention` within company (non‑overlapping).
- Re‑estimate overall and split by tone at the event week: `sentiment_score ≥ 0` vs `< 0`.
5) Cross‑correlation (lead–lag) check
- Create L1–L4 lags for predictors and compute company‑level cross‑correlations of `NYT_mention` vs `num_memes_z` to visualize lead/lag structure.
6) Placebo design (matched on NYT intensity)
- For each positive/negative event, select a placebo week within the same company, outside ±`w`, matched on `NYT_mention` decile (rank-based). If no match, relax to any week outside ±`w`.
- Build placebo event windows and compute the same outcome.
7) Event − Placebo difference
- Compute `Diff(τ) = Mean_event(τ) − Mean_placebo(τ)` with 95% CIs via SE aggregation; report as main robustness result.

## Defaults and settings
- Window: `w = 3` weeks.
- Percentile threshold for tone events: 90th within company.
- Descriptive time‑series smoothing: 4‑week moving average (for display only).
- RNG seed for placebo selection: 42.
- Equal weighting across events (no brand re‑weighting).

## Key outputs (paths)
- Descriptive time series per brand (examples):
  - `figures/ts_<brand>_num_articles_vs_num_memes_z.png`
- Main event‑study CIs (NaN‑aware z outcome):
  - `figures/event_pos_num_memes_z_ci.png`
  - `figures/event_neg_num_memes_z_ci.png`
- Alternative outcomes (relative, log1p):
  - `figures/event_pos_num_memes_rel_ci.png`, `figures/event_neg_num_memes_rel_ci.png`
  - `figures/event_pos_log1p_meme_volume_ci.png`, `figures/event_neg_log1p_meme_volume_ci.png`
  - `figures/event_pos_log1p_meme_engagement_ci.png`, `figures/event_neg_log1p_meme_engagement_ci.png` (if engagement exists)
- Non‑overlap, alignment, and demeaned diagnostics:
  - `figures/event_pos_num_memes_z_ci_nooverlap.png`, `figures/event_neg_num_memes_z_ci_nooverlap.png`
  - `figures/event_pos_num_memes_z_ci_shift_m1.png`, `figures/event_pos_num_memes_z_ci_shift_p1.png`
  - `figures/event_pos_num_memes_z_demeaned_ci.png`, `figures/event_neg_num_memes_z_demeaned_ci.png`
- Mentions‑spike events (overall and tone‑split):
  - `figures/event_mentions_num_memes_z_ci.png`
  - `figures/event_mentions_pos_num_memes_z_ci.png`, `figures/event_mentions_neg_num_memes_z_ci.png`
- Cross‑correlation figure:
  - `figures/xcorr_NYT_mention_vs_num_memes_z.png`
- Event − Placebo differences (main robustness):
  - `figures/results/event_diff/event_pos_num_memes_z_diff_ci.png`
  - `figures/results/event_diff/event_neg_num_memes_z_diff_ci.png`

## Interpretation guide (at a glance)
- Positive (negative) tone events capture unusually positive (negative) news coverage for a brand; the plots show average meme activity before/after those weeks.
- A significant positive value at τ=0 or τ>0 suggests meme activity contemporaneous with or following news tone spikes; τ<0 patterns indicate pre‑trends.
- Placebo‑matched differences help separate tone‑specific effects from generic activity associated with high news intensity.


