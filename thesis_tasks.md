# Feladatlista a diplomamunkához
## Mi van meg eddig?
- A `build_panel.ipynb` és a belőle származó heti panel (pl. `company_weekly_panel_normalized.csv`) már tartalmazza a NYT-hírek heti darabszámát (num_articles), sentiment-átlagait (sentiment_score, mean_pos, mean_neg) és a mémek darabszámát (num_memes).
- A `lead_lag_analysis.ipynb`/`.py`-ben küldtél cross-korrelációs és event-study ábrákat, amelyek megmutatták, hogy a hírek pozitív/negatív tartalma hol vezet, hol követi a mém aktivitást.
- Az `enriched_predictions_metadata.csv`-ben szerepel posztszintű engagement (score → `engagement`), valamint `num_comments` és `total_awards_received`,ezeket sem vontad még be a panelbe.
  
## Mi hiányzik még a mostani adatbázisból?

A "panel" kifejezés azt jelenti, hogy több különböző brand adatait figyeljük meg, több héten keresztül. Ez a fajta adatstruktúra lehetővé teszi, hogy megvizsgáljuk, hogyan változnak a dolgok az idő múlásával az egyes brandeknél.
- **Meme_sentiment** – Jelenleg csak a NYT-cikkek hangulata (sentiment) szerepel az adatbázisban, de a mémek hangulata még hiányzik.
- **Meme_engagement** – A mémekhez kapcsolódó felhasználói aktivitás (pl. like-ok, megosztások) heti szinten még nincs összesítve.
- **NYT_mention_{b,t}** – A NYT-cikkek heti darabszáma jelenleg `num_articles` néven szerepel, de a professzor átnevezné `NYT_mention_{b,t}`-re, ahol `b` a brandet és `t` az adott hetet jelöli.
- **Kontroll változók és fixhatások** – Ezek további tényezők, amik befolyásolhatják a mém aktivitást, de nem a fő célunk vizsgálni őket. Fontos őket figyelembe venni, hogy pontosabb képet kapjunk a hírek és mémek közötti kapcsolatról.
**
## 1. Új adatok, amiket létre kell hozni (brand b, hét t)
Ezek az új adatok segítenek jobban megérteni a mémek és a hírek közötti kapcsolatot:
- **Meme_volume_{b,t}**: Adott brand `b` adott héten `t` közzétett mém-posztjainak száma.
- **Meme_sentiment_{b,t}**: Adott brand `b` mém-posztjainak átlagos hangulata (sentiment) az adott héten `t`.
- **Meme_engagement_{b,t}**: Adott brand `b` mém-posztjainak átlagos felhasználói aktivitása (engagement, pl. upvotes összeg, amit esetleg érdemes logaritmikus skálán nézni: log(1+upvotes)) az adott héten `t`.
- **NYT_mention_{b,t}**: Adott brand `b` NYT-cikkeinek száma az adott héten `t`.
- **NYT_sentiment_{b,t}**: Adott brand `b` NYT-cikkeinek átlagos hangulata (sentiment) az adott héten `t`.
- **További jellemzők**: Gondolkodjunk el, hogy van-e más, hasznos jellemző, például a mémek témakategóriája.
**
## 2. Időegység
**Heti aggregálás (t = év×hét)**: Az összes adatot heti szintre kell összesíteni, azaz minden egyes brand minden egyes hetére egyetlen értéket kapunk a fenti változókból.

## 3. Lagolt változók
**Lagolt magyarázók**: Ezek a változók az előző hetek (t-1, t-2, stb.) értékeit jelentik. Például, ha a NYT_mention_{b,t-1}-et nézzük, az azt jelenti, hogy az előző heti NYT említések számát vizsgáljuk. Ezek segítenek abban, hogy lássuk, a múltbeli események hogyan befolyásolják a jelenlegi mém aktivitást. Már van egy `make_lags` függvényed, amit ehhez használhatsz.

## 4. Kontroll változók és Fixhatások
Ezek olyan statisztikai módszerek, amelyekkel kizárhatjuk a zavaró tényezők hatását, így jobban láthatjuk a hírek és mémek közötti valódi kapcsolatot.
- **Brand fixhatás (γ_b)**: Ez azt jelenti, hogy figyelembe vesszük az egyes brandek egyedi, időben állandó jellemzőit, amelyek befolyásolhatják a mém aktivitást (pl. egy brand általános ismertsége, népszerűsége). Így nem tévesztjük össze ezeket a brand-specifikus különbségeket a hírek hatásával.
- **Idő fixhatás (δ_t, év×hét)**: Ez segít kiszűrni azokat az általános időbeli trendeket vagy szezonális mintákat (pl. karácsonyi időszak, nyári uborkaszezon), amelyek minden brandre egyformán hatnak. Ezzel biztosítjuk, hogy az eredményeink ne az általános időbeli változások miatt torzuljanak.
- **Szezonális dummyk**: Ezek specifikus változók, amelyek bizonyos szezonális hatásokat (pl. negyedévek, hónapok) modelleznek.
- **Általános hírliget (top100 NYT) trend**: Ez a legnépszerűbb 100 NYT cikk általános trendjét jelenti, ami segíthet a szélesebb médiafigyelem hatásának kiszűrésében.

## 5. Leíró elemzések (Visualizációk)
Ezek az ábrák segítenek vizuálisan megérteni az adatok közötti kapcsolatokat:
- **Idősorok**: Ábrázoljuk az idő függvényében a **Meme_volume** és **NYT_mention** változásait egy adott brandre vagy az összesített top100 brandre.
- **Kereszt-korrelációs függvény (CCF)**: Ez egy grafikon, ami megmutatja, hogy a **Meme_volume** és **NYT_mention** hogyan mozognak együtt, és van-e köztük valamilyen késleltetett (lead/lag) kapcsolat. Már csináltál ilyet.
- **Korrelációs mátrix + szórásdiagramok**: Ezek részletesebb képet adnak a változók közötti kapcsolatokról:
  - **NYT_sentiment** vs **Meme_sentiment**: Hogyan függ össze a NYT cikkek és a mémek hangulata?
  - **NYT_mention** vs **Meme_volume**: Hogyan függ össze a NYT említések száma és a mémek volumene?
- **Egyszerű esemény-specifikus (event-stacked) grafikonok**: Ez a fajta grafikon azt mutatja meg, hogyan változik a **Meme_volume** vagy **Meme_engagement** egy nagy esemény (pl. egy fontos hír) előtt és után (pl. ±30 nap). Ezt már megcsináltad.

## 6. Formális modell (Panel Regresszió)
Ez a statisztikai modell segít számszerűsíteni a hírek hatását a mémekre, miközben figyelembe veszi a fent említett kontrollokat és fixhatásokat.
```math
Y_{b,t} = \beta_0 + \beta_1\,NYT\_mention_{b,t-\ell} + \beta_2\,NYT\_sentiment_{b,t-\ell} + \gamma_b + \delta_t + X_{b,t} + \varepsilon_{b,t}
```
- **Y_{b,t} (Kimenet/Függő változó)**: Azt méri, amit magyarázni akarunk, pl. a *log(1 + Meme_volume)* (a mémek száma logaritmikus skálán) vagy a *Meme_sentiment* (mémek hangulata) az adott `b` brandnél az `t` héten.
- **\beta_0**: Az alapérték (konstans).
- **\beta_1, \beta_2**: Ezek az együtthatók (koefficiensek) azt mutatják meg, hogy a `NYT_mention` és `NYT_sentiment` változók mennyire befolyásolják a `Y_{b,t}`-t. A `-ℓ` azt jelenti, hogy ezeknek a változóknak a korábbi heti (`lagolt`) értékeit használjuk.
- **\gamma_b (Brand fixhatás)**: Ahogy fent említettük, ez kezeli azokat a brand-specifikus dolgokat, amik időben állandóak.
- **\delta_t (Idő fixhatás)**: Ez kezeli azokat az általános időbeli trendeket, amik minden brandre hatnak.
- **X_{b,t} (Kontroll változók)**: Ide tartoznak a további kontrollváltozók, mint például a subreddit aktivitás vagy az általános Reddit aktivitás.
- **\varepsilon_{b,t} (Hiba tag)**: Ez reprezentálja azokat a tényezőket, amiket a modell nem magyaráz meg.
- **Standard hibák klaszterezése brand szerint**: Ez egy módszer a statisztikai hibák (standard errorok) kiszámítására, ami figyelembe veszi, hogy az ugyanazon brandhez tartozó adatok valószínűleg jobban hasonlítanak egymásra, mint a különböző brandek adatai. Ez pontosabb eredményeket ad.

## 7. További tesztek
- **Granger-okozatisági teszt**: Ez a teszt statisztikailag segít eldönteni, hogy a hírek változásai előrejelzik-e a mémek változásait.
- **Esemény-specifikus (Event study) konfidenciaintervallumokkal**: A már elkészült esemény-specifikus grafikonokat egészítsük ki konfidenciaintervallumokkal, hogy lássuk, statisztikailag szignifikánsak-e a megfigyelt változások az események körül.

## Hogyan illeszkednek a javaslatok a meglévő eredményekhez?
- A javasolt új változók kibővítik a már működő kereszt-korrelációs (CCF) elemzéseket: nemcsak volument, hanem sentimentet és engagementet is lehet párosítani a NYT-adatokkal.
- A lagolt változók ötlete ugyanaz, amit a `make_lags` függvényeddel már megcsináltál (max_lag=4); a professzor csak explicit listázta, hogy ezt minden új változóra is alkalmazd.
- A formalizált panel regressziós modell (Y_{b,t}= …) természetes folytatása a korrelációs ábráknak: az ábrák vizuálisan jeleztek kapcsolatot, a modell ezt számszerűsíti és kontrolálja a fixhatásokkal.
- A Granger-okozatiság és az esemény-specifikus konfidenciaintervallumok továbbviszik a már elkészített event-stacked grafikonjaidat analitikusabb irányba (szignifikancia-teszt, idősoros okság).
- A `event_*_meme_spike_raw/rel/z.png` ábrák jó vizuális alapot szolgáltatnak, de konfidenciaintervallumok nélkül; ezért kéri a professzor a konfidenciaintervallumokat és egy regresszió alapú esemény-specifikus elemzést.

## 8. Prioritás – Kötelező feladatok vs. ötletek

### Kötelező (minimálisan teljesítendő)
- Új változók létrehozása (Meme_volume, Meme_sentiment, Meme_engagement, NYT_mention, NYT_sentiment) heti szinten.
- Lagolt változók képzése a fenti változókra (t-1, t-2, …).
- Leíró idősor- és CCF-ábrák elkészítése.
- Panel regressziós modell becslése fixhatásokkal és lagolt magyarázókkal, brand-szintű klaszterezett hibákkal.
- Esemény-specifikus (event-study) konfidenciaintervallumok számítása a már meglévő event-stacked grafikonokhoz.

### Opcionális / továbbfejleszthető
- További jellemzők a mémekhez (pl. témakategória, alternatív sentiment-mérő).
- Kontrollváltozók bővítése: subreddit fixhatások, szezonális dummyk, általános NYT-trend (top100).
- Granger-okozatisági teszt a hírek → mémek irányra.
- Engagement log-transzformáció (log(1+upvotes)) robusztus vizsgálata.
- További vizualizációk (korrelációs mátrix, scatterplotok) publikációra kész minőségben.