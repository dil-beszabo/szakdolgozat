# Lead–Lag Analysis: módszerek és fő választások

Ez a leírás röviden, közérthetően összefoglalja, mit és hogyan csinál a `code/lead_lag_analysis.ipynb` notebook. 
Cél: mit jelentenek az ábrák, és mire figyelj értelmezéskor.

## Adat és előfeldolgozás

- Forrás panel: `data/panels/company_weekly_panel_analysis_ready.csv` (kiegyensúlyozott, naptár szerint teljes heti panel).
- Időkulcs: `week_start` (heti frekvencia); csoportosítás: `company`.

## Kimenetek (outcome-ok)

- Fő kimenet: `num_memes_z_es` (NaN‑aware z‑score a heti meme volume‑ra vállalaton belül).
  - Ha egy héten nincs megfigyelés (pl. `num_memes == 0` és nincs sentiment), azt NaN‑nak kezeljük az átlagoláshoz.
  - A z‑score vállalaton belül, csak a megfigyelt heteken számolt átlag és szórás alapján készül.
  - A rajzoknál alias: `num_memes_z = num_memes_z_es`.
- Robusztus alternatívák:
  - `num_memes_rel` (vállalaton belüli relatív normalizálás).
  - `log1p_meme_volume = log1p(num_memes)`.
  - `log1p_meme_engagement = log1p(meme_engagement)` (ha van engagement).

## Események és ablak (window)

- Pozitív esemény: a vállalat `mean_pos` értéke a heti eloszlás felső 10%-ában (90. percentilis felett).
- Negatív esemény: a vállalat `mean_neg` értéke a heti eloszlás felső 10%-ában.
- Eseményablak: `w = 3` hét, azaz τ ∈ {−3, −2, −1, 0, +1, +2, +3}.

## Ábrázolás és bizonytalanság

- Minden esemény körül (τ szerint) átlagoljuk a kiválasztott kimenetet.
- 95% konfidencia‑sáv: `mean ± 1.96 * (sd / sqrt(n))` τ‑onként, események felett.
- Megjegyzés: ez egyszerű, pontonkénti (pointwise) sáv, nem kezeli külön a sorfüggést vagy a többszörös tesztelést.

## Fő eredmények

- Idősorok (márkánként): `NYT_mention` vs `num_memes_z` időbeli együttmozgás.
- Event‑study (95% CI): pozitív/negatív tone események körüli átlagos `num_memes_z` alakulása.
- Event − Placebo különbség: esemény és párosított placebo (azonos vállalat, ±w‑n kívül, azonos `NYT_mention` decil) különbsége τ‑onként, 95% CI‑vel.
- Cross‑correlation (lead–lag): korreláció `NYT_mention` (L1–L4) és `num_memes_z` között a késés/vezetés mintázatának feltárására.

## Diagnosztikák és robusztaság

Rövid magyarázat:
- Non‑overlap: ugyanannál a vállalatnál az események között legalább `w` hét szünetet tartunk. Cél: ne fedjék egymást az ablakok.
- Alignment shift: az esemény indexét −1 vagy +1 héttel eltoljuk. Cél: ellenőrizni az időzítés érzékenységét.
- Week‑demeaned: az adott hét vállalati átlagát kivonjuk a kimenetből. Cél: közös heti sokkok kiszűrése.
- Mentions‑spike: esemény a `NYT_mention` vállalati eloszlásának felső 10%-a (nem a tone).
- Tone split: a mentions‑spike eseményeket megbontjuk az eseményhéten mért `sentiment_score` előjele szerint (≥ 0 vs < 0).

## Alapbeállítások

- Ablak: `w = 3` hét.
- Eseményküszöb (tone): vállalaton belüli 90. percentilis.
- Idősor simítás (leíró ábrák): 4 hetes mozgóátlag (csak megjelenítés).
- RNG seed (placebo kiválasztás): 42.
- Események egyenlő súllyal (nincs vállalati súlyozás).

## Fő outputok (útvonalak)

- Leíró idősor (márkánként):
  - `figures/ts_<brand>_num_articles_vs_num_memes_z.png`
- Fő event‑study CIk (NaN‑aware z kimenet):
  - `figures/event_pos_num_memes_z_ci.png`
  - `figures/event_neg_num_memes_z_ci.png`
- Alternatív kimenetek (relative, log1p):
  - `figures/event_pos_num_memes_rel_ci.png`, `figures/event_neg_num_memes_rel_ci.png`
  - `figures/event_pos_log1p_meme_volume_ci.png`, `figures/event_neg_log1p_meme_volume_ci.png`
  - `figures/event_pos_log1p_meme_engagement_ci.png`, `figures/event_neg_log1p_meme_engagement_ci.png` (ha van engagement)
- Non‑overlap, alignment shift, week‑demeaned (lásd a fenti magyarázatot):
  - `figures/event_pos_num_memes_z_ci_nooverlap.png`, `figures/event_neg_num_memes_z_ci_nooverlap.png`
  - `figures/event_pos_num_memes_z_ci_shift_m1.png`, `figures/event_pos_num_memes_z_ci_shift_p1.png`
  - `figures/event_pos_num_memes_z_demeaned_ci.png`, `figures/event_neg_num_memes_z_demeaned_ci.png`
- Mentions‑spike (összesítve és tone‑split):
  - `figures/event_mentions_num_memes_z_ci.png`
  - `figures/event_mentions_pos_num_memes_z_ci.png`, `figures/event_mentions_neg_num_memes_z_ci.png`
- Cross‑correlation:
  - `figures/xcorr_NYT_mention_vs_num_memes_z.png`
- Event − Placebo különbségek (fő robusztaság):
  - `figures/results/event_diff/event_pos_num_memes_z_diff_ci.png`
  - `figures/results/event_diff/event_neg_num_memes_z_diff_ci.png`

## Rövid értelmezési segédlet

- Pozitív/negatív tone‑esemény: szokatlanul pozitív/negatív hírhangnem az adott héten.
- τ=0 vagy τ>0 pozitív eltérés: a meme‑aktivitás együtt mozog vagy követi a hírsokkot.
- τ<0 mintázat: előzetes trend (pre‑trend) gyanúja.
- Placebo‑különbség: segít elválasztani a tone‑specifikus hatást az általános hírsűrűséghez kötődő aktivitástól.

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
5) Placebo design (NYT intensity alapján párosítva)
   - Minden pozitív/negatív eventhez választunk egy placebo hetet ugyanazon vállalaton belül, ±`w`‑n kívül, `NYT_mention` decil (rank‑based) szerint párosítva. Ha nincs találat, lazítunk: bármely hét ±`w`‑n kívül.
   - Placebo event window‑k felépítése és ugyanazon outcome kiszámítása.
6) Event − Placebo difference
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


