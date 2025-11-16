# Lead–Lag Analysis: módszerek és fő választások

Ez a jegyzet összefoglalja a `code/lead_lag_analysis.ipynb` notebookban megvalósított módszertani választásokat, amelyekkel az ábrákat előállítjuk és az eredményeket értelmezzük.

## Adat és előfeldolgozás

- Forrás panel: `data/panels/company_weekly_panel_analysis_ready.csv` (kiegyensúlyozott, naptár‑teljes heti panel).
- Időkulcs: `week_start` (heti frekvencia); csoportosítási kulcs: `company`.

## Outcome konstrukció

- Fő outcome: vállalatonkénti z‑score a meme volume‑ra, NaN‑aware kezelés a valódi nem‑megfigyelésre:
  - Ha `num_memes == 0` és (opcionálisan) hiányzik a `mean_meme_sentiment`, az átlagolásnál tekintsük nem megfigyeltnek (NaN).
  - A vállalati átlagot és szórást csak a megfigyelt heteken számoljuk; az értékeket z‑score‑oljuk `num_memes_z_es` néven.
  - Ábra/legacy alias: `num_memes_z = num_memes_z_es`.
- Robusztasági outcome‑ok:
  - `num_memes_rel` (relatív normalizálás panel utilokból).
  - `log1p_meme_volume = log1p(num_memes)`.
  - `log1p_meme_engagement = log1p(meme_engagement)`, ha van engagement.

## Event definíció és windowing

- Positive events: olyan company‑week, amely a vállalaton belüli `mean_pos` 90. percentilise felett van.
- Negative events: olyan company‑week, amely a vállalaton belüli `mean_neg` 90. percentilise felett van.
- Event window: szimmetrikus `w = 3` hét, azaz τ ∈ {−3,…,0,…,+3}.
- Ha külön nem jelezzük, az átfedő események megengedettek; diagnosztikához non‑overlap kikényszerítése min. `w` hétnyi távolsággal.

## Estimation és bizonytalanság
- Minden eventhez összegyűjtjük az outcome‑ot minden τ‑nál, majd eventek között átlagolunk (egyenlő súllyal).
- Uncertainty bands: pointwise 95% CIs a `mean ± 1.96 * (sd / sqrt(n))` képlettel, τ‑nként az eventek felett.
- Megjegyzés: a CIs független event window‑kat feltételez, nem korrigál serial correlationre vagy multiple testingre (szándékosan egyszerű).

## Diagnostics és robustness checks
1) Non‑overlap constraint
   - Az eseményeket újraépítjük min. `w` hétnyi távolsággal, és újra‑becsüljük az event study‑kat.
2) Alignment shift tests
   - A pozitív eseményeket −1 és +1 héttel eltoljuk; újra‑becslés az igazítás ellenőrzésére (τ=0 válasz gyengül, ha rossz az alignment).
3) Week‑demeaned outcome
   - A `num_memes_z` értékeket heti keresztmetszeti átlaggal demeaneljük a közös heti sokkok elnyelésére, majd újra‑becslünk.
4) Mentions‑spike events és tone split
   - Esemény: a vállalaton belüli `NYT_mention` top 10% (non‑overlapping).
   - Újra‑becslés összesítve és tone szerinti bontásban az event héten: `sentiment_score ≥ 0` vs `< 0`.
5) Cross‑correlation (lead–lag) check
   - L1–L4 lagok a prediktorokra; vállalati szintű cross‑correlation `NYT_mention` és `num_memes_z` között a lead/lag struktúra vizualizálásához.
6) Placebo design (NYT intensity alapján párosítva)
   - Minden pozitív/negatív eventhez választunk egy placebo hetet ugyanazon vállalaton belül, ±`w`‑n kívül, `NYT_mention` decil (rank‑based) szerint párosítva. Ha nincs találat, lazítunk: bármely hét ±`w`‑n kívül.
   - Placebo event window‑k felépítése és ugyanazon outcome kiszámítása.
7) Event − Placebo difference
   - `Diff(τ) = Mean_event(τ) − Mean_placebo(τ)` 95% CIs‑szel (SE aggregáció); ezt jelentjük fő robusztasági eredményként.

## Alapbeállítások
- Window: `w = 3` hét.
- Percentilis küszöb tone eventekhez: 90th vállalaton belül.
- Leíró idősor simítás: 4 hetes moving average (csak megjelenítés).
- RNG seed placebo kiválasztáshoz: 42.
- Egyenlő súlyozás az eventek között (nincs brand re‑weighting).

## Fő outputok (útvonalak)
- Leíró idősor márkánként (példák):
  - `figures/ts_<brand>_num_articles_vs_num_memes_z.png`
- Fő event‑study CIs (NaN‑aware z outcome):
  - `figures/event_pos_num_memes_z_ci.png`
  - `figures/event_neg_num_memes_z_ci.png`
- Alternatív outcome‑ok (relative, log1p):
  - `figures/event_pos_num_memes_rel_ci.png`, `figures/event_neg_num_memes_rel_ci.png`
  - `figures/event_pos_log1p_meme_volume_ci.png`, `figures/event_neg_log1p_meme_volume_ci.png`
  - `figures/event_pos_log1p_meme_engagement_ci.png`, `figures/event_neg_log1p_meme_engagement_ci.png` (ha van engagement)
- Non‑overlap, alignment és demeaned diagnosztikák:
  - `figures/event_pos_num_memes_z_ci_nooverlap.png`, `figures/event_neg_num_memes_z_ci_nooverlap.png`
  - `figures/event_pos_num_memes_z_ci_shift_m1.png`, `figures/event_pos_num_memes_z_ci_shift_p1.png`
  - `figures/event_pos_num_memes_z_demeaned_ci.png`, `figures/event_neg_num_memes_z_demeaned_ci.png`
- Mentions‑spike events (overall és tone‑split):
  - `figures/event_mentions_num_memes_z_ci.png`
  - `figures/event_mentions_pos_num_memes_z_ci.png`, `figures/event_mentions_neg_num_memes_z_ci.png`
- Cross‑correlation ábra:
  - `figures/xcorr_NYT_mention_vs_num_memes_z.png`
- Event − Placebo különbségek (fő robustness):
  - `figures/results/event_diff/event_pos_num_memes_z_diff_ci.png`
  - `figures/results/event_diff/event_neg_num_memes_z_diff_ci.png`

## Rövid értelmezési útmutató
- Positive (negative) tone events: szokatlanul pozitív (negatív) hírhangnem a márkáról; az ábrák az átlagos meme activity‑t mutatják az esemény előtti/utáni hetekben.
- Szignifikáns pozitív érték τ=0‑nál vagy τ>0‑nál: a meme activity együtt mozog vagy követi a news tone spike‑okat; τ<0 minták pre‑trends‑re utalnak.
- Placebo‑matched különbségek segítenek elválasztani a tone‑specifikus hatásokat a magas news intensity‑hez kapcsolódó általános aktivitástól.


