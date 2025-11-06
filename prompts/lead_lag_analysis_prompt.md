# NYT → Meme Lead–Lag Analysis 

This plan builds weekly company-level features from NYT sentiment and meme activity, then tests whether negative or positive articles precede meme spikes and which direction is stronger. 
## Definitions in context
1. Meme timestamp field to use: 
- `created_utc` in `predictions_manifest.csv`
1. Meme spike definition: 
- weekly num_memes > mean + 2σ within company 
## Steps 
1. Build NYT weekly sentiment per company
	- Parse `data/nyt/*.txt` (already implemented in `code/process_nyt_articles.py`).
	- Compute per-week: mean_pos, mean_neu, mean_neg, sentiment_score (pos−neg), non_neutral_share, num_articles. 
	- Save to `data/derived/nyt_weekly_sentiment.csv`. 
2. Build meme weekly activity per company from `data/prediction_images/`. 
	- Read `predictions_metadata.csv`. 
	- Normalize company names (lowercase, alnum; e.g., `MercedesBenz`→`mercedes`). 
	- Compute per-week per company: num_memes, (optional) mean_meme_score if available. 
	- Create `meme_spike` indicator by chosen rule (spike if weekly num_memes > mean + 2σ within company). 
	- Save to `data/derived/memes_weekly_activity.csv`. 
3. Join into a weekly panel - Inner-join on (company, date); forward-fill gaps optional. 
	- Create lagged NYT features L1..L4: sentiment_score, mean_pos, mean_neg, non_neutral_share, num_articles.
	- Save `data/derived/company_weekly_panel.csv`. 
4. Cross-correlation (lead–lag) 
	- For each company, compute correlations *corr(NYT_t−k, meme_spike_t)* for k=1..4, and *corr(NYT_t, meme_spike_t+k)* for k=1..4. 
	- Summarize across companies (median, IQR). Plot correlation vs lag for: sentiment_score, mean_pos, mean_neg. 
5. Event study (directional strength) 
	- Define positive events: top 10% of weekly mean_pos; negative events: top 10% of weekly mean_neg. 
	- For each event type, compute average meme outcomes over windows t∈[−3,+3]: meme_spike rate, num_memes. 
	- Compare areas or peak responses to assess which direction is stronger. 
6. Optional robustness 
	- Use z-scores within company for NYT features. 
	- Logistic model: meme_spike_t ~ L1..L3 of sentiment_score, mean_pos, mean_neg, num_articles, company FE.
## Outputs 
- `nyt_weekly_sentiment.csv`, `memes_weekly_activity.csv`, `company_weekly_panel.csv`. 
- Plots: lagged correlations, event-study response curves.
## Minimal code artifacts 
- `code/build_weekly_panel.py`: produce CSVs above. 
- `code/lead_lag_analysis.py`: compute correlations/event-study and save plots.

