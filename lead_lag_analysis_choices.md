# Lead–Lag Analysis: Methods and Key Choices\n
\n
This note summarizes the methodological choices implemented in `code/lead_lag_analysis.ipynb` to generate the figures and interpret results.\n
\n
## Data and preprocessing\n
\n
- Source panel: `data/panels/company_weekly_panel_analysis_ready.csv` (balanced, calendar-complete weekly panel).\n
- Time key: `week_start` (weekly frequency); grouping key: `company`.\n
- Aliases used for backward compatibility:\n
  - `nyt_pos_share` → `mean_pos`\n
  - `nyt_neg_share` → `mean_neg`\n
  - `nyt_sentiment` → `sentiment_score`\n
  - `nyt_non_neutral_share` → `non_neutral_share`\n
  - `NYT_mention` also serves as `num_articles` where needed.\n
\n
## Outcome construction\n
\n
- Main outcome: per‑company z‑score of meme volume with NaN-aware handling of true non-observation:\n
  - If `num_memes == 0` and (optionally) `mean_meme_sentiment` is missing, treat as unobserved (NaN) when averaging.\n
  - Compute per‑company mean and sd on observed weeks only; z‑score into `num_memes_z_es`.\n
  - Alias for plotting/legacy helpers: `num_memes_z = num_memes_z_es`.\n
- Alternative outcomes used in robustness:\n
  - `num_memes_rel` (relative normalization from panel utilities).\n
  - `log1p_meme_volume = log1p(num_memes)`.\n
  - `log1p_meme_engagement = log1p(meme_engagement)` if engagement is present.\n
\n
## Event definition and windowing\n
\n
- Positive events: company‑week at or above the 90th percentile of `mean_pos` within the company.\n
- Negative events: company‑week at or above the 90th percentile of `mean_neg` within the company.\n
- Event window: symmetric `w = 3` weeks, i.e., τ ∈ {−3,…,0,…,+3}.\n
- Unless stated otherwise, overlapping events are allowed; for diagnostics we enforce non‑overlap with a minimum gap of `w`.\n
\n
## Estimation and uncertainty\n\n
- For each event, collect the outcome at each τ, then average across events (equal weight per event).\n
- Uncertainty bands: pointwise 95% CIs using `mean ± 1.96 * (sd / sqrt(n))` across events at each τ.\n
- Note: CIs treat event windows as independent and do not adjust for serial correlation or multiple testing (kept simple by design).\n
\n
## Diagnostics and robustness checks\n\n
1) Non‑overlap constraint\n\n
- Rebuild events with a minimum distance of `w` weeks and re‑estimate the event studies.\n\n
2) Alignment shift tests\n\n
- Shift identified positive events by −1 and +1 week; re‑estimate to verify alignment (τ=0 response should attenuate when misaligned).\n\n
3) Week‑demeaned outcome\n\n
- Demean `num_memes_z` by the cross‑sectional weekly mean to absorb common weekly shocks and re‑estimate.\n\n
4) Mentions‑spike events and tone split\n\n
- Define events as top 10% of `NYT_mention` within company (non‑overlapping).\n
- Re‑estimate overall and split by tone at the event week: `sentiment_score ≥ 0` vs `< 0`.\n\n
5) Cross‑correlation (lead–lag) check\n\n
- Create L1–L4 lags for predictors and compute company‑level cross‑correlations of `NYT_mention` vs `num_memes_z` to visualize lead/lag structure.\n\n
6) Placebo design (matched on NYT intensity)\n\n
- For each positive/negative event, select a placebo week within the same company, outside ±`w`, matched on `NYT_mention` decile (rank-based). If no match, relax to any week outside ±`w`.\n
- Build placebo event windows and compute the same outcome.\n\n
7) Event − Placebo difference\n\n
- Compute `Diff(τ) = Mean_event(τ) − Mean_placebo(τ)` with 95% CIs via SE aggregation; report as main robustness result.\n
\n
## Defaults and settings\n\n
- Window: `w = 3` weeks.\n
- Percentile threshold for tone events: 90th within company.\n
- Descriptive time‑series smoothing: 4‑week moving average (for display only).\n
- RNG seed for placebo selection: 42.\n
- Equal weighting across events (no brand re‑weighting).\n
\n
## Key outputs (paths)\n\n
- Descriptive time series per brand (examples):\n
  - `figures/ts_<brand>_num_articles_vs_num_memes_z.png`\n
- Main event‑study CIs (NaN‑aware z outcome):\n
  - `figures/event_pos_num_memes_z_ci.png`\n
  - `figures/event_neg_num_memes_z_ci.png`\n
- Alternative outcomes (relative, log1p):\n
  - `figures/event_pos_num_memes_rel_ci.png`, `figures/event_neg_num_memes_rel_ci.png`\n
  - `figures/event_pos_log1p_meme_volume_ci.png`, `figures/event_neg_log1p_meme_volume_ci.png`\n
  - `figures/event_pos_log1p_meme_engagement_ci.png`, `figures/event_neg_log1p_meme_engagement_ci.png` (if engagement exists)\n
- Non‑overlap, alignment, and demeaned diagnostics:\n
  - `figures/event_pos_num_memes_z_ci_nooverlap.png`, `figures/event_neg_num_memes_z_ci_nooverlap.png`\n
  - `figures/event_pos_num_memes_z_ci_shift_m1.png`, `figures/event_pos_num_memes_z_ci_shift_p1.png`\n
  - `figures/event_pos_num_memes_z_demeaned_ci.png`, `figures/event_neg_num_memes_z_demeaned_ci.png`\n
- Mentions‑spike events (overall and tone‑split):\n
  - `figures/event_mentions_num_memes_z_ci.png`\n
  - `figures/event_mentions_pos_num_memes_z_ci.png`, `figures/event_mentions_neg_num_memes_z_ci.png`\n
- Cross‑correlation figure:\n
  - `figures/xcorr_NYT_mention_vs_num_memes_z.png`\n
- Event − Placebo differences (main robustness):\n
  - `figures/results/event_diff/event_pos_num_memes_z_diff_ci.png`\n
  - `figures/results/event_diff/event_neg_num_memes_z_diff_ci.png`\n
\n
## Interpretation guide (at a glance)\n\n
- Positive (negative) tone events capture unusually positive (negative) news coverage for a brand; the plots show average meme activity before/after those weeks.\n
- A significant positive value at τ=0 or τ>0 suggests meme activity contemporaneous with or following news tone spikes; τ<0 patterns indicate pre‑trends.\n
- Placebo‑matched differences help separate tone‑specific effects from generic activity associated with high news intensity.\n
\n

