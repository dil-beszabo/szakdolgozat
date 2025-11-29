# Panel Analysis Workflow: módszerek és fő választások

Ez a dokumentum összefoglalja a code/panel_analysis_workflow.ipynb notebookban alkalmazott módszertani választásokat, valamint az eredmények értelmezéséhez szükséges technikai háttérinformációkat.

A notebook a következő elemzési lépéseket tartalmazza:

- márkaszintű leíró idősorok,
- event-study elemzések (pozitív/negatív tone események hatása),
- cross-correlation (lead–lag) elemzés,
- kiterjedt robusztasági diagnosztikák és sanity checkek.

## Adat és előfeldolgozás

- Forrás panel: `data/panels/company_weekly_panel_analysis_ready.csv` (kiegyensúlyozott, minden vállalat-hét kombinációt tartalmazó panel).
- Időkulcs: `week_start` (heti frekvencia); csoportosítási kulcs: `company`.

A panel a builder modul outputja, amely tartalmazza a NYT-eredetű hangulatelemeket (mean_pos, mean_neg, sentiment_score stb.), a meme-aktivitási változókat, valamint laggelt és normalizált transzformációkat.

## Outcome konstrukció

- Fő outcome: vállalatonkénti z‑score a meme volume‑ra, NaN‑aware kezelés a valódi nem‑megfigyelésre:
  - Ha `num_memes == 0` és (opcionálisan) hiányzik a `mean_meme_sentiment`, az átlagolásnál tekintsük nem megfigyeltnek (NaN).
  - A vállalati átlagot és szórást csak a megfigyelt heteken számoljuk; az értékeket z‑score‑oljuk `num_memes_z_es` néven.
  - Ábra/legacy alias: `num_memes_z = num_memes_z_es`.
- Robusztasági outcome‑ok:
  - `num_memes_rel` - vállalaton belüli relatív normalizálás (8 hetes, visszatekintő rolling mean).
  - `log1p_meme_volume = log1p(num_memes)`.
  - `log1p_meme_engagement = log1p(meme_engagement)`, ha van engagement.

## Event definíció és windowing

- Positive events: olyan company‑week, amely a vállalaton belüli `mean_pos` 90. percentilise felett van.
- Negative events: olyan company‑week, amely a vállalaton belüli `mean_neg` 90. percentilise felett van.
- Event window: szimmetrikus `w = 3` hét, azaz τ ∈ {−3,…,0,…,+3}.
  
## Estimation és bizonytalanság
- Minden eseményhez kivágjuk a τ-ablakot, majd τ-onként az események között átlagolunk (egyenlő súlyokkal).
- Uncertainty bands: pointwise 95% CIs a `mean ± 1.96 * (sd / sqrt(n))` képlettel, τ‑nként az eventek felett.

## Diagnosztikák és robusztaság

1) **Non‑overlap constraint**
   - Az eseményeket újraépítjük min. `w` hétnyi távolsággal, és újra‑becsüljük az event study‑kat.
   - Cél: ne fedjék egymást az ablakok.

2) **Alignment shift tests**
   - A pozitív eseményeket −1 és +1 héttel eltoljuk; újra‑becslés az igazítás ellenőrzésére (τ=0 válasz gyengül, ha rossz az alignment).
   - Cél: ellenőrizni az időzítés érzékenységét.

3) **Week‑demeaned outcome**
   - Minden vállalat-hét outcome értékéből levonjuk az adott hét keresztmetszeti átlagát (összes vállalat átlaga azon a héten):
     ```
     num_memes_z_dm = num_memes_z − week_mean(num_memes_z)
     ```
   - Cél: közös heti sokkok kiszűrése (pl. platformszintű változások, szezonalitás, általános internetes aktivitási hullámok).
   - Ha az event-study minta a demeaned outcome-mal is megmarad, az esemény hatása valóban vállalat-specifikus.

4) **Mentions‑spike events és tone split**
   - Esemény: a vállalaton belüli `NYT_mention` top 10% (non‑overlapping).
   - Újra‑becslés összesítve és tone szerinti bontásban az event héten: `sentiment_score ≥ 0` vs `< 0`.

5) **Placebo design** (NYT intensity alapján párosítva)
  - Minden valódi tone-esemény mellé választunk egy placebo hetet:
     - ugyanazon vállalaton belül,
     - ±w héten kívül,
     - megegyező NYT_mention decillel (rank-alapú),
     - ha nincs találat → bármely ±w-n kívüli hét.
   - Ugyanazt az event-study eljárást alkalmazzuk a placebo ablakokra is.
6) **Event − Placebo difference**
  - A különbség: `Diff(τ) = Mean_event(τ) − Mean_placebo(τ)` 95% CI-vel.

## Alapbeállítások
- Window: `w = 3` hét.
- Percentilis küszöb tone eventekhez: 90th vállalaton belül.
- Leíró idősor simítás: 4 hetes moving average (csak megjelenítés).
- RNG seed placebo kiválasztáshoz: 42.
- Egyenlő súlyozás az eventek között (nincs brand re‑weighting).

## Fő outputok (útvonalak)
- Márka-szintű idősorok:
  - `figures/ts_<brand>_num_articles_vs_num_memes_z.png`
- Fő event‑study CI eredmények (NaN‑aware z outcome):
  - `figures/event_pos_num_memes_z_ci.png`
  - `figures/event_neg_num_memes_z_ci.png`
- Alternatív outcome‑ok:
  - `figures/event_pos_num_memes_rel_ci.png`, `figures/event_neg_num_memes_rel_ci.png`
  - `figures/event_pos_log1p_meme_volume_ci.png`, `figures/event_neg_log1p_meme_volume_ci.png`
  - `figures/event_pos_log1p_meme_engagement_ci.png`, `figures/event_neg_log1p_meme_engagement_ci.png` (ha van engagement)
- Diagnosztikák
  - Non‑overlap, alignment és demeaned diagnosztikák:
    - `figures/event_pos_num_memes_z_ci_nooverlap.png`, `figures/event_neg_num_memes_z_ci_nooverlap.png`
    - `figures/event_pos_num_memes_z_ci_shift_m1.png`, `figures/event_pos_num_memes_z_ci_shift_p1.png`
    - `figures/event_pos_num_memes_z_demeaned_ci.png`, `figures/event_neg_num_memes_z_demeaned_ci.png`
  - Mentions‑spike events (overall és tone‑split):
    - `figures/event_mentions_num_memes_z_ci.png`
    - `figures/event_mentions_pos_num_memes_z_ci.png`, `figures/event_mentions_neg_num_memes_z_ci.png`
  - Event − Placebo különbségek (fő robustness):
    - `figures/results/event_diff/event_pos_num_memes_z_diff_ci.png`
    - `figures/results/event_diff/event_neg_num_memes_z_diff_ci.png`
- Cross‑correlation ábra:
  - `figures/xcorr_NYT_mention_vs_num_memes_z.png`
  
