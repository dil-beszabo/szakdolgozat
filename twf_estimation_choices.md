## TWFE estimation: key modeling choices (for interpreting results)

### What we estimate

- **Outcomes**: `log1p_meme_volume`, `mean_meme_sentiment`, `log1p_meme_engagement`.
- **Main predictors**: `NYT_mention` and, in some specifications, `nyt_sentiment`.
- **Lag structure**: We include the current week (k = 0) and lags \(k \in \{1,2,3,4\}\). We run robustness with fewer lags (3, 2, 1).
- **Fixed effects (TWFE)**: brand fixed effects `C(company)` and calendar-week fixed effects `C(week_fe)` to control for time‑invariant brand heterogeneity and common shocks.
- **Estimator**: OLS via formula API with two‑way fixed effects included as dummies.
- **Standard errors**: clustered by `company`.

In short, for the “with sentiment” specification we estimate:

$$
Y_{b,t} = \alpha_b + \delta_t + \sum_{k=0}^{K} \beta_k\,\mathrm{NYT\_mention}_{b,t-k} + \sum_{k=0}^{K} \theta_k\,\mathrm{nyt\_sentiment}_{b,t-k} + \varepsilon_{b,t}
$$

and for “mention‑only” we omit the \(\theta_k\) terms.

### Why create mention-only:
Sentiment scores are only defined when the New York Times published at least one article about the brand in that week, sentiment coverage is sparse (~7% of brand–weeks). To retain comparability, two model families are estimated:

1. **Mention-only models**, using all brand–weeks, where `NYT_mention_Lk` = 0 indicates absence of coverage.
2. **Sentiment-including models**, restricted to brand–weeks where sentiment is observed for the current and previous weeks.
 
### Data and sample

- **Panel**: weekly brand panel loaded from `data/panels/company_weekly_panel_analysis_ready.csv`.
- **Week FE key**: `week_fe = week_start.strftime("%Y-%W")`.
- **Missing data**: rows with any missing in the outcome, FE keys, or included predictors (current + required lags) are dropped per outcome/spec.
  - As a result, “with sentiment” specs have smaller N than “mention‑only”.

### Specifications we run

- **With sentiment (main)**: includes `NYT_mention` and `nyt_sentiment` with current + lags.
- **Mention‑only (robustness/availability)**: includes `NYT_mention` only with current + lags.
- **Lag windows**: K ∈ {4, 3, 2, 1}. For each K we save the current and lag coefficients for plotting/synthesis.

### Inference and tests

- **SEs**: company‑clustered, robust to arbitrary serial correlation and heteroskedasticity within a brand.
- **Joint lag tests**: Wald tests on the sum of lag coefficients (e.g., \(\sum_{k=1}^{K} \beta_k = 0\)) are computed and printed to assess cumulative delayed effects. (Applied for mentions in both variants; sentiment lags tested in the “with sentiment” runs.)

### Interpretation guidance

- **Fixed effects**: Coefficients are identified off within‑brand, across‑week variation relative to brand means and common week shocks.
- **Log1p outcomes**: For volume/engagement, coefficients are approximately semi‑elasticities; small coefficients can be read as approximate percent changes in the outcome given a one‑unit change in the predictor.
- **Lags**: `NYT_mention_Lk` (and `nyt_sentiment_Lk`) capture associations k weeks after the news signal, conditional on current/other lags and FE.
- **Cumulative effects**: Use the reported lag‑sum tests and summed coefficients to discuss total delayed response.

### Outputs produced

- We export key coefficients (current + each lag) for each outcome and lag window to CSV under `data/derived/`:
  - `twfe_key_coefficients_with_sentiment_{K}_lags.csv`
  - `twfe_key_coefficients_mention_only_{K}_lags.csv`

These files are the source for the coefficient path/lag plots and summary tables in the results section.


