## Egyszerű útmutató: Fixhatásos panel regresszió (mémek × NYT)

Ez az útmutató röviden és példákkal mutatja meg, hogyan futtasd le a fixhatásos panel regressziót a már elkészített heti brand–hét panelen.

### Mit szeretnénk megtudni?
- A NYT említések száma és a NYT cikkek hangulata (korábbi hetekben) előre jelzi-e:
  - a mémek heti számát (Meme_volume),
  - a mémek átlagos hangulatát (Meme_sentiment),
  - a mémek átlagos engagementjét (Meme_engagement)?

### A te panel fájlod
- Fájl: `data/derived/company_weekly_panel_analysis_ready.csv`
- Kulcsok: `company`, `week_start`
- Fő változók:
  - Kimenetek (Y): `log1p_meme_volume`, `mean_meme_sentiment`, `log1p_meme_engagement`
  - NYT magyarázók (X): `NYT_mention`, `nyt_sentiment`, és lags: `NYT_mention_L1..L4`, `nyt_sentiment_L1..L4`
  - Fixhatásokhoz hasznos: `iso_year`, `iso_week` (vagy közvetlenül `week_start`)

### Modell (szavakkal és képlettel)
Szavakkal: a jelen heti mém-aktivitást (Y) a NYT említések és a NYT hangulat korábbi heti értékei magyarázzák, miközben kiszűrjük a márka-szintű állandó különbségeket és a minden márkára közös heti hatásokat.

$$
Y_{b,t} =
  \beta_0 + \sum_{\ell=1}^{L} \beta_{1\ell} \, \mathrm{NYT\_mention}_{b,t-\ell}
  + \sum_{\ell=1}^{L} \beta_{2\ell} \, \mathrm{nyt\_sentiment}_{b,t-\ell}
  + \gamma_b + \delta_t + \varepsilon_{b,t}
$$

- `b`: brand (márka), `t`: hét
- `γ_b`: brand fixhatás (márkánként állandó különbségek)
- `δ_t`: hét fixhatás (időbeli/szezonális közös hatások)
- Tipikusan \(L = 1..4\) lagot használunk (ezek benne vannak a CSV-ben).

### Mit jelentenek az együtthatók?
- `β1ℓ > 0`: az ℓ héttel ezelőtti több NYT-említés → most több mém / nagyobb engagement / pozitívabb mém-hangulat.
- `β2ℓ > 0`: az ℓ héttel ezelőtti pozitívabb NYT-hangulat → most több mém / nagyobb engagement / pozitívabb mém-hangulat.
- Hasznos megnézni a lag-együtthatók összegét is:

$$
\sum_{\ell=1}^{L} \beta_{1\ell}
$$

és

$$
\sum_{\ell=1}^{L} \beta_{2\ell}
$$

Ez azt mutatja meg, hogy a teljes késleltetett NYT-említések vagy NYT-hangulat kumulatív hatása mennyi.

---``

## Gyors gyakorlati példa (Python/statsmodels)

### 1) Betöltés
```python
import pandas as pd
panel = pd.read_csv(
    \"/Users/beszabo/bene/szakdolgozat/data/derived/company_weekly_panel_analysis_ready.csv\",
    parse_dates=[\"week_start\"],
)

# Heti fixhatás azonosító (kategória kulcs)
panel[\"week_fe\"] = panel[\"week_start\"].dt.strftime(\"%Y-%W\")
```

### 2) Alapmodell: Y = log1p_meme_volume, csak L1 lagokkal
```python
import statsmodels.formula.api as smf

df = panel.dropna(subset=[\"log1p_meme_volume\", \"NYT_mention_L1\", \"nyt_sentiment_L1\"]).copy()

mod = smf.ols(
    \"log1p_meme_volume ~ NYT_mention_L1 + nyt_sentiment_L1 + C(company) + C(week_fe)\",
    data=df,
).fit(cov_type=\"cluster\", cov_kwds={\"groups\": df[\"company\"]})  # hibák klaszterezése brand szerint

print(mod.summary())
```

Értelmezés: ha `NYT_mention_L1` koefficiense pozitív és szignifikáns, akkor az előző heti több NYT-cikk átlagosan magasabb idei heti mém-volument jelez.

### 3) Bővített modell: L1–L4 lagok + kumulatív hatások tesztje
```python
cols = [
    \"NYT_mention_L1\", \"NYT_mention_L2\", \"NYT_mention_L3\", \"NYT_mention_L4\",
    \"nyt_sentiment_L1\", \"nyt_sentiment_L2\", \"nyt_sentiment_L3\", \"nyt_sentiment_L4\",
]

need = [\"log1p_meme_volume\", \"company\", \"week_fe\"] + cols
df = panel.dropna(subset=need).copy()

formula = (
    \"log1p_meme_volume ~ \"
    + \" + \".join(cols)
    + \" + C(company) + C(week_fe)\"
)

mod = smf.ols(formula, data=df).fit(cov_type=\"cluster\", cov_kwds={\"groups\": df[\"company\"]})
print(mod.summary())

# Kumulatív hatás: összes NYT_mention lag összege
test_mentions = mod.t_test(\"NYT_mention_L1 + NYT_mention_L2 + NYT_mention_L3 + NYT_mention_L4 = 0\")
print(\"Sum(NYT_mention_L1..L4):\", test_mentions)

# Kumulatív hatás: összes nyt_sentiment lag összege
test_sent = mod.t_test(\"nyt_sentiment_L1 + nyt_sentiment_L2 + nyt_sentiment_L3 + nyt_sentiment_L4 = 0\")
print(\"Sum(nyt_sentiment_L1..L4):\", test_sent)
```

### 4) Alternatív kimenetek (ugyanaz a minta)
- Mém-hangulat:
```python
y = \"mean_meme_sentiment\"
df = panel.dropna(subset=[y, \"NYT_mention_L1\", \"nyt_sentiment_L1\", \"week_fe\"]).copy()
mod = smf.ols(f\"{y} ~ NYT_mention_L1 + nyt_sentiment_L1 + C(company) + C(week_fe)\", data=df).fit(
    cov_type=\"cluster\", cov_kwds={\"groups\": df[\"company\"]}
)
print(mod.summary())
```

- Mém-engagement (log1p):
```python
y = \"log1p_meme_engagement\"
df = panel.dropna(subset=[y, \"NYT_mention_L1\", \"nyt_sentiment_L1\", \"week_fe\"]).copy()
mod = smf.ols(f\"{y} ~ NYT_mention_L1 + nyt_sentiment_L1 + C(company) + C(week_fe)\", data=df).fit(
    cov_type=\"cluster\", cov_kwds={\"groups\": df[\"company\"]}
)
print(mod.summary())
```

---

## Mit kell lejelenteni?
- Koeficiensek és p-értékek a NYT-lagokra (külön `NYT_mention` és `nyt_sentiment`).
- Kumulatív lag-hatások (összeg) és azok tesztje.
- Fixhatások jelenléte: `C(company)`, `C(week_fe)` szerepel a képletben.
- Standard hibák: klaszterezve brand szerint.
- N (megfigyelések), márkák száma (klaszterek), R² (within is jó, de a sima OLS summary is elfogadható).

## Gyors ellenőrző lista
- [ ] A megfelelő Y változót választottad (volume / sentiment / engagement)?
- [ ] Benne van a képletben a `C(company)` és `C(week_fe)`?
- [ ] Klaszterezett hibát kértél: `cov_type=\"cluster\", groups=company`?
- [ ] L1 vagy L1–L4 lagok szerepelnek a modellben?
- [ ] Kiszámoltad a lag-koefficiensek összegét és tesztjét?

Tippek:
- Ha túl sok a hiányzó érték, kezdd L1-gyel és csak később bővíts L4-ig.
- `log1p_meme_volume` és `log1p_meme_engagement` stabilabb lehet, mint a nyers értékek.
- Ha különösen zajos, nézd meg az egyes márkák idősort is (diagnosztika).


