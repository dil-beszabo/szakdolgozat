# <mark style="background: #FF5582A6;">Topic relevancy</mark>
....

# <mark style="background: #FFB8EBA6;">Building the dataset</mark>
....


# <mark style="background: #ABF7F7A6;">Building the OCR</mark>
....

# <mark style="background: #BBFABBA6;">Building the classifier</mark>
....

# <mark style="background: #FFF3A3A6;">Using CLIP instead</mark>
....

# <mark style="background: #FFB86CA6;">Getting the distribution</mark>
....

# <mark style="background: #ADCCFFA6;">Fetching NYT data</mark>
Bar diagram of num_of_memes and num_of_articles per companies
![bar_chart](https://github.com/dil-beszabo/szakdolgozat/blob/main/company_num_articles.png)
![bar_chart](https://github.com/dil-beszabo/szakdolgozat/blob/main/company_num_memes.png)



| Top companies by memes: | num_memes | Top companies by articles: | num_articles |
| ----------------------- | --------- | -------------------------- | ------------ |
| Apple                   | 331       | Facebook                   | 1990         |
| Google                  | 323       | Youtube                    | 1884         |
| Youtube                 | 282       | Netflix                    | 1830         |
| Microsoft               | 240       | Google                     | 1591         |
| Nintendo                | 125       | Instagram                  | 1560         |
| Instagram               | 113       | Apple                      | 1475         |
| Netflix                 | 99        | Microsoft                  | 1348         |
| Amazon                  | 89        | Amazon                     | 1243         |
| Facebook                | 80        | Tesla                      | 1030         |
| Disney                  | 78        | Spotify                    | 917          |

# <mark style="background: #D2B3FFA6;">Context Engineering</mark>
A `topicality-online` kutatas repojat felhasznalva, a `lead_lag_analysis_prompt.md` prompt megirasa tortent, amivel ....

# <mark style="background: #CACFD9A6;">Why FinBert</mark>
Certain terms can have multiple meaning, we need them in the financial context

## Adding aliases
By incorporating synonyms and alternative spellings for company names, the analysis can now:
1. Capture More Relevant Data: Articles that previously might have been missed because they used an alias instead of the canonical company name are now included. This leads to a more comprehensive dataset for sentiment analysis.
2. Increase Signal Strength: With more relevant articles contributing to the sentiment scores, the "signal" of positive or negative news events becomes clearer and often more pronounced. This translates to stronger peaks and deeper troughs in the event study plots.
3. Reinforce Lead-Lag Patterns: The overall patterns of how meme activity responds to news events (e.g., meme spikes often preceding or following news sentiment) are generally reinforced, indicating that the alias logic is helping to identify these relationships more accurately, rather than distorting them.
## Normalise meme volumes 

A mémek számának normalizálása, mint például a z-pontszám (num_memes_z) további segítséget nyújt. Ez azért fontos, mert a különböző cégek mémaktivitása eltérő lehet (pl. egy kis cég 10 mémje nagy, egy nagy cégnek pedig kevés). A normalizálás lehetővé teszi, hogy a mémaktivitás csúcsait és mélypontjait objektíven hasonlítsuk össze a cégek között, és jobban azonosítsuk az igazi "tüskéket" (spike-okat).
- Z-score (num_memes_z): Azt mutatja meg, hogy egy adott heti mémaktivitás mennyire tér el egy cég saját történelmi átlagától, standard deviációban kifejezve. Ez segít összehasonlítani a kiemelkedés nagyságát különböző cégek között, függetlenül attól, hogy melyiknek van alapvetően magasabb mémaktivitása.

# <mark style="background: #FF5582A6;">Results</mark>

Heti bontásban minden céghez kiszámoltam az átlagos memeszámot és a heti hírek sentimentjét.  
A híreket minden cégnél a pozitív (vagy negatív) cikkek átlagos aránya szerint rangsoroltam.  
A legpozitívabb (illetve legnegatívabb) 10%-ba tartozó hetek alkotják az eseményhalmazt, ahol  
τ = 0 jelöli a „legpozitívabb” vagy „legnegatívabb” hetet.

A diagramok azt mutatják, hogyan változott a cégekről készült mémek normalizált (z-score) száma az esemény hete körül, a cégek saját átlagukhoz viszonyítva.

![bar_chart](https://github.com/dil-beszabo/szakdolgozat/blob/main/event_pos_num_memes_z.png)

![bar_chart](https://github.com/dil-beszabo/szakdolgozat/blob/main/event_neg_num_memes_z.png)

Miután pozitív híreket hoztak le egy cégről több meme készült róla, negatív esetben pedig kevesbb.

