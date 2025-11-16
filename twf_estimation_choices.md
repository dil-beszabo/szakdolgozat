## TWFE estimation: fő modellezési választások (az eredmények értelmezéséhez)

### Mit becslünk

- **Kimenetek**: `log1p_meme_volume`, `mean_meme_sentiment`, `log1p_meme_engagement`.
- **Fő prediktorok**: `NYT_mention`, és egyes specifikációkban `nyt_sentiment`.
- **Kontroll (baseline)**: késleltetett subreddit activity `subreddit_activity_{t-1}` (heti poszt+komment a brand‑releváns subreddit(ek)ben).
- **Késleltetési szerkezet**: tartalmazzuk az aktuális hetet (k = 0) és a késleltetéseket \(k \in \{1,2,3,4\}\). Robusztaságként kevesebb késleltetéssel (3, 2, 1) is futtatunk.
- **Fixed effects (TWFE)**: brand fixed effects `C(company)` és calendar-week fixed effects `C(week_fe)`, hogy kontrolláljuk az időben állandó márka‑heterogenitást és a közös heti sokkokat.
- **Estimator**: OLS a formula API‑n, two‑way fixed effects dummyként felvéve.
- **Standard errors**: clustered by `company`.

Röviden, a “with sentiment” specifikációban a következőt becsüljük:

$$
Y_{b,t} = \alpha_b + \delta_t + \sum_{k=0}^{K} \beta_k\,\mathrm{NYT\_mention}_{b,t-k} + \sum_{k=0}^{K} \theta_k\,\mathrm{nyt\_sentiment}_{b,t-k} + \lambda\,\mathrm{subreddit\_activity}_{b,t-1} + \varepsilon_{b,t}
$$

és a “mention‑only” esetben a \(\theta_k\) tagokat elhagyjuk.

### Miért van mention‑only:
`nyt_sentiment` csak akkor definiált, ha a New York Times az adott héten legalább egy cikket publikált a márkáról; a sentiment lefedettség ritka (~7% a brand–hetek között). Az összehasonlíthatóság érdekében két modellcsaládot becslünk:

1. **Mention‑only modellek**, az összes brand–hetet használva, ahol `NYT_mention_Lk` = 0 a lefedettség hiányát jelzi.
2. **Sentiment‑et tartalmazó modellek**, azokra a brand–hetekre korlátozva, ahol a sentiment a tárgyhéten és a szükséges megelőző heteken megfigyelhető.
 
### Adat és minta

- **Panel**: heti brand panel betöltve innen: `data/panels/company_weekly_panel_analysis_ready.csv`.
- **Week FE key**: `week_fe = week_start.strftime("%Y-%W")`.
- **Subreddit activity**: a `subreddit_activity` a brand‑releváns subreddit(ek) heti posztolási és kommentelési volumenét méri; ha több brand‑specifikus subreddit létezik, márka szinten aggregáljuk. Mivel a datasetben egy subreddit van, ez a változó a heti platform‑forgalom ingadozásait tükrözi, amelyek korrelálhatnak a mémaktivitással.
- **Missing data**: azokat a sorokat eldobjuk, ahol a kimenetben, a FE kulcsokban vagy a bevont prediktorokban (aktuális + szükséges késleltetések) hiányzó érték van.
  - Ennek következtében a “with sentiment” specifikációk kisebb N‑nel futnak, mint a “mention‑only”.

### Futtatott specifikációk

- **With sentiment (main)**: tartalmazza a `NYT_mention` és `nyt_sentiment` változókat az aktuális héttel és késleltetésekkel.
- **Mention‑only (robustness/availability)**: csak `NYT_mention` az aktuális héttel és késleltetésekkel.
- **Lag windows**: \(K \in \{4, 3, 2, 1\}\). Minden K‑ra elmentjük a current és lag koefficienseket ábrázoláshoz/összegzéshez.
- **Kontrollok**: a baseline tartalmazza a `subreddit_activity_{t-1}` változót.
- **Seasonality and global trend**: `C(week_fe)` jelenlétében elhagyjuk a month/holiday dummykat és a globális NYT‑trendet; ezeket csak `C(week_fe)` nélküli robusztasági variánsokban használjuk. Mivel a calendar‑week FE elnyeli az aggregált sokkokat és a globális hírintenzitás trendjeit, a további dummyk (month, holidays) és a globális NYT‑mention összegek \(\delta_t\)‑vel tökéletesen kollineárisak lennének, ezért elhagyjuk őket.

### Következtetés és tesztek

- **SEs**: company‑clustered; robusztus tetszőleges brand‑szintű serial correlation és heteroskedasticity mellett.
- **Joint lag tests**: Wald tests a lag koefficiensek összegére (pl. \(\sum_{k=1}^{K} \beta_k = 0\)) a kumulatív késleltetett hatások értékelésére. (Alkalmazva a mentions esetén mindkét variánsban; a sentiment késleltetések a “with sentiment” futásokban vannak tesztelve.)

### Értelmezési útmutató

- **Fixed effects**: a koefficiensek az egyes márkákon belüli, hetek közötti variációból azonosítottak, a márkaátlagokhoz és a közös heti sokkokhoz viszonyítva.
- **Log1p outcomes**: volume/engagement esetén a koefficiensek közelítőleg semi‑elasticities; kis értékek nagyjából százalékos változásként olvashatók a kimenetben egy egységnyi prediktor‑változásra.
- **Lags**: a `NYT_mention_Lk` (és `nyt_sentiment_Lk`) k héttel a hírsokkok után fennálló asszociációkat ragadnak meg, a current/más késleltetések és a FE feltételével.
- **Cumulative effects**: a jelentett lag‑sum teszteket és az összegzett koefficienseket használd a teljes késleltetett hatás megvitatásához.
- **Dynamics**: a baseline OLS nem tartalmaz lagged outcomes; a dinamikus panel variánsok (pl. Arellano–Bond) csak opcionális robusztaságként szerepelnek. A `subreddit_activity_{t-1}` kontroll a márka szintű, időben változó platform‑forgalmat ragadja meg, és késleltetve szerepel a bad‑control kockázat mérséklésére.

### Előállított outputok

- Minden kimenetre és lag ablakra exportáljuk a fő koefficienseket (current + minden lag) CSV‑be a `data/derived/` alá:
  - `twfe_key_coefficients_with_sentiment_{K}_lags.csv`
  - `twfe_key_coefficients_mention_only_{K}_lags.csv`

Ezek a fájlok szolgálnak a koefficiens‑pálya/lag ábrák és az összefoglaló táblák forrásaként az eredmények szekcióban.


