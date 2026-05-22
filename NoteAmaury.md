# Notes de travail — Étude événementielle sentiment × indicateurs de marché

> Ce document retrace chronologiquement les étapes de réflexion et les résultats
> de l'analyse. Les scripts correspondants sont dans `KPI_analysis/` ; les plots
> dans `KPI_analysis/output/plots/sentiment_summary/`.

---

## 1. Exploration préliminaire : rendements ajustés autour des publications

L'idée de départ est de construire, pour un ticker donné, une série de rendements ajustée du biais sectoriel, puis de vérifier visuellement si les dates de publication des rapports annuels coïncident avec des mouvements anormaux du titre.

### Construction du rendement ajusté

Pour un stock $i$, on part du prix de clôture journalier $P_{i,t}$ et on calcule le rendement simple :

$$
r_{i,t} = \frac{P_{i,t} - P_{i,t-1}}{P_{i,t-1}}
$$

On récupère ensuite l'ensemble des tickers de la même industrie et on calcule la moyenne des rendements :

$$
\bar{r}_{\text{industrie}, t} = \frac{1}{N} \sum_{j=1}^{N} r_{j,t}
$$

Le rendement "sans biais sectoriel" est alors :

$$
r^{\text{adj}}_{i,t} = r_{i,t} - \bar{r}_{\text{industrie}, t}
$$

### Dates de publication

Les dates de publication des rapports annuels sont récupérées depuis EDGAR (`acceptanceDateTime`), ramenées à une fréquence journalière :

$$
d^{\text{publication}} = \lfloor t^{\text{publication}} \rfloor_{\text{jour}}
$$

### Première impression

On superpose ces dates à trois séries (rendement ajusté, rendement brut vs industrie, prix). Résultat plutôt décevant : les publications ne produisent pas de rupture nette visible. Cela ne veut pas dire qu'il n'y a pas d'effet, mais qu'il faut une approche plus structurée.

**Hypothèse** : les résultats peuvent être neutres ou déjà anticipés par le marché. Il faudrait labéliser chaque rapport (bon / mauvais / neutre) pour conditionner l'analyse.

---

## 2. Classification du sentiment des lettres CEO par LLM

Pour labéliser chaque rapport, on a extrait les lettres CEO (cf. `doc_text_processing/CEO_word_extraction/`) puis on les a classifiées en **positive / negative / neutral** via un LLM servi localement avec vLLM (Qwen3.5-9B, structured output JSON).

### Premier prompt : tout positif

Un premier prompt naïf produisait presque exclusivement des labels "positive". Ce n'est pas surprenant : les lettres CEO sont systématiquement rédigées sur un ton optimiste, même lors de mauvaises années. Le modèle captait le ton plutôt que le fond.

### Prompt corrigé

On a construit un prompt plus directif qui force le modèle à ignorer le ton et à se concentrer sur les chiffres rapportés :

> *"You are a financial analyst evaluating a company's annual performance. Read the CEO letter below and classify the company's FINANCIAL RESULTS for that year. Ignore the optimistic tone — CEO letters are always written positively, even in bad years. Focus strictly on the reported numbers: revenue, earnings, profit, margins, cash flow, debt. Pay attention to hedging language: if the CEO emphasizes future improvement, recovery, or 'turning a corner', that usually signals the current year was poor. Phrases like 'despite challenges', 'headwinds', 'difficult environment', or 'we expect to do better' are red flags — classify as negative unless hard numbers clearly show growth.*
>
> *Rules:*
> - *positive: revenue or earnings grew year-over-year, margins improved, or clear financial progress backed by numbers*
> - *negative: revenue or earnings declined, losses reported, impairments, restructuring charges, deteriorating metrics, OR the letter is mostly forward-looking promises without reporting actual good numbers*
> - *neutral: genuinely mixed results with some metrics clearly up and others clearly down, backed by specific numbers*
>
> *When in doubt between neutral and negative, lean negative."*

Résultat : répartition plus réaliste (46 positive, 25 negative, 0 neutral sur les 71 lettres disponibles — les 157 paires ticker/année sans lettre CEO sont ignorées).

---

## 3. Grandeurs étudiées

On construit les indicateurs suivants pour chaque stock $i$ au jour $t$ :

### Rendements

$$r_{i,t} = \frac{P_{i,t} - P_{i,t-1}}{P_{i,t-1}}$$

$$r^{\text{adj}}_{i,t} = r_{i,t} - \bar{r}_{\text{industrie}, t}$$

### Volatilité (écart-type glissant 20 jours des rendements)

$$\sigma_{i,t} = \text{std}_{20}(r_{i,\cdot})$$

$$\sigma^{\text{adj}}_{i,t} = \sigma_{i,t} - \bar{\sigma}_{\text{industrie}, t}$$

### Volume ATS (volume normalisé par la moyenne historique du titre)

$$V^{\text{ATS}}_{i,t} = \frac{V_{i,t}}{\bar{V}_i}$$

$$V^{\text{adj}}_{i,t} = V^{\text{ATS}}_{i,t} - \bar{V}^{\text{ATS}}_{\text{industrie}, t}$$

### Prix normalisé (pour event studies, J0 = 1)

$$\tilde{P}_{i,d} = \frac{P_{i,t_0 + d}}{P_{i,t_0}}$$

où $t_0$ est le jour de publication et $d \in [-10, +10]$ jours de trading.

Au total, 7 grandeurs initiales : rendement, volatilité, volume (chacune en brut et non biaisé) + prix normalisé.

---

## 4. Trois types de visualisation

1. **Bar charts** (fenêtre fixe) : pour chaque grandeur, moyenne sur `WINDOW=5` jours après la publication, groupée par sentiment. Barres d'erreur = SEM. Vue agrégée : "en moyenne, quand le sentiment est négatif, le return/volume/volatilité est-il différent ?"

2. **Event studies** (dynamique temporelle) : pour chaque grandeur, moyenne jour par jour de J-10 à J+10 avec bande de confiance à 95%. Permet de voir si un pic/creux apparaît au moment de la publication, et s'il y a anticipation ou réaction retardée.

3. **Distributions** : histogrammes des valeurs par document pour chaque sentiment × métrique, avec mean/median. Permet de visualiser la forme des distributions et d'identifier les outliers.

---

## 5. Résultats des event studies agrégées (n=169 positive, n=153 negative)

### Rendements cumulés

- **`cum_return` (raw)** : avant J0, les titres "negative" ont un rendement cumulé positif (~+2% à J-10) qui converge vers 0 à J0. Les titres "positive" restent plats (~0%). Après J0, les deux groupes décrochent légèrement (−1% à −1.5% à J+10). Pas de séparation nette post-publication entre les deux sentiments — les CI se chevauchent largement.

- **`cum_return_unbiased` (vs industrie EW)** : même pattern — les titres négatifs surperforment leur industrie avant J0 (+1 à +2%), puis convergent. Post-publication, les deux groupes oscillent autour de 0 sans tendance claire. Les CI se chevauchent.

- **`cum_return_unbiased_vw` (vs industrie VW)** : pattern similaire avec une surperformance pre-event plus prononcée pour les négatifs (+2% à J-10). Post-J0, les deux courbes s'effondrent vers 0 et se superposent.

**Interprétation** : les résultats négatifs semblent avoir été **anticipés par le marché** dans les 10 jours précédant la publication (surperformance pré-event = correction de la sous-évaluation antérieure ?). Le jour de la publication lui-même ne génère pas de mouvement additionnel significatif en rendement.

### Volatilité

- **`norm_raw_volatility` (% change from J0)** : les deux sentiments présentent exactement le même pattern — une hausse progressive de la volatilité après J0 (+15–20% à J+10 vs J0). Aucune différence entre positive et negative. La volatilité augmente systématiquement après la publication, indépendamment du contenu.

- **`norm_unbiased_volatility` (% change from J0)** : ici la séparation est forte. Les titres **positifs** voient leur unbiased volatility monter fortement après J0 (jusqu'à +500% à J+10), avec un CI très large. Les négatifs restent plats (~0%). Cependant, les **niveaux pré-event** sont très différents : les positifs partent de −4 (soit 400% en dessous de J0) tandis que les négatifs partent de −1. Ce pattern asymétrique à CI large suggère quelques outliers parmi les positifs plutôt qu'un signal robuste.

### Volume

- **`norm_raw_volume` (% change from J0)** : pic de volume à J+1 pour les **négatifs** (+40%) qui retombe rapidement. Les positifs montrent un pic moindre (+20% à J+1). Les CI se chevauchent dès J+2. Le sur-volume à la publication est plus marqué pour les mauvais résultats.

- **`norm_unbiased_volume` (% change from J0)** : très bruité, pas de pattern clair. Les CI sont énormes (±10 à ±15), les moyennes oscillent autour de 0 sans tendance. Non exploitable en l'état.

- **`norm_volume_unbiased_vw` (% change from J0)** : extrêmement bruité (échelle ±15). Le calcul souffre du problème de division par ~0 à J0 (cf. §8). Non exploitable.

### Synthèse — tableau de signal

**Lecture** : toutes les métriques portent un signal de publication plus ou moins fort. La colonne *Signal J0* qualifie si le jour de publication est visible (forme du graphe) ; la colonne *Discrimination ±* indique si les courbes positive et negative se séparent.

#### Métriques en niveau (mean)

| Métrique | Signal J0 | Force | Discrimination ± | Détail |
|---|---|---|---|---|
| `raw_volume` (Volume ATS mean) | **Très fort** — rampe des deux côtés vers un pic net à J0/J+1 (+50% vs baseline) | ★★★ | Faible — les deux sentiments montent ensemble | Publication day est l'événement le plus propre du dataset |
| `raw_volatility` (Stock Volatility mean) | **Fort** — inflexion nette à J0, puis montée monotone post-J0 | ★★☆ | Faible — negative légèrement > positive en niveau absolu, mais même dynamique | Les deux groupes montent en parallèle |
| `unbiased_volatility` (mean) | **Fort** — saut discret (step function) à J0 | ★★☆ | **Oui** — positive > negative en unbiased (moins négatif) ; les deux sautent mais à des niveaux séparés | Confirme que la vol propre au titre augmente au jour de publication |
| `unbiased_volume` (mean) | **Fort** — pic à J0/J+1, retour ensuite | ★★☆ | **Oui** — positive systématiquement au-dessus de negative (écart ~0.2–0.3 pre-J0, persiste post-J0) | Volume propre des positifs > négatifs |
| `volume_unbiased_vw` (mean) | **Fort** — pic à J0/J+1 | ★★☆ | **Oui** — positive > negative, même pattern que unbiased_volume | Confirmation VW du résultat EW |
| `raw_price_norm` (J0=1) | **Modéré** — convergence des deux courbes vers 1.0 à J0 depuis des niveaux pré-J0 différents | ★★☆ | Pré-J0 uniquement — positive part au-dessus (~1.008), negative en dessous (~0.997). Post-J0, déclin parallèle | L'info est intégrée à J0 |
| `cum_return` | **Modéré** — convergence vers 0 à J0 | ★★☆ | Pré-J0 uniquement — negative surperforme de +2% avant J0, puis rejoint positive | Anticipation pré-publication par le marché |
| `cum_return_unbiased` | **Modéré** — convergence à J0 | ★★☆ | Pré-J0 uniquement — même pattern | EW ne change pas la conclusion |
| `cum_return_unbiased_vw` | **Modéré** — convergence à J0, un peu plus net que EW | ★★☆ | Pré-J0 uniquement — negative surperforme de +2.3% avant J0 | VW légèrement plus propre |

#### Métriques normalisées (% change from J0)

| Métrique | Signal J0 | Force | Discrimination ± | Détail |
|---|---|---|---|---|
| `norm_raw_volatility` | **Fort** — palier plat pré-J0, montée monotone post-J0 (+15–20% à J+10) | ★★☆ | Très faible — les deux courbes se superposent | La forme confirme J0 comme point d'inflexion |
| `norm_raw_volume` | **Fort** — spike à J+1 | ★★☆ | **Oui** — negative +70% à J+1 vs positive +20% | Les mauvaises nouvelles génèrent plus de sur-volume |
| `norm_unbiased_volatility` | Visible mais bruité — positive explose (+500%), negative reste plat | ★☆☆ | Apparent mais non fiable — CI très large, outliers probables | Division par $\approx 0$ amplifie le bruit |
| `norm_unbiased_volume` | Non exploitable — bruit pur | ☆☆☆ | Non | CI de ±10, aucun pattern |
| `norm_volume_unbiased_vw` | Non exploitable | ☆☆☆ | Non | Même problème de division par ~0 |

### Conclusion

**1. Le jour de la publication est visible sur toutes les métriques en niveau.** Le signal est universel — il n'y a pas de métrique brute qui ne montre pas J0. Classement par force :
- **Volume** (★★★) : pic franc, le plus propre, pas de CI overlap à J0.
- **Volatilité, rendements cumulés, prix** (★★☆) : inflexion / convergence nette, mais les CI restent proches.

**2. La discrimination positive / negative est partielle et dépend de la métrique :**
- **Volume (unbiased, raw normalized)** : discrimine — les négatifs génèrent plus de sur-volume à J+1.
- **Volatilité unbiased** : discrimine en niveau — les positifs ont une vol propre plus haute (moins corrigée par l'industrie).
- **Rendements cumulés** : discrimine uniquement en pré-event — les négatifs surperforment avant J0, pas après. Le marché anticipe.
- **Volatilité raw, prix normalisé** : pas de discrimination — les deux sentiments évoluent en parallèle.

**3. Les métriques normalisées non biaisées** (norm_unbiased_*) sont inexploitables à cause de la division par $\approx 0$ à J0.


## 6. Biais de sélection sur la courbe "Industry avg" (volatilité)

Sur les event study par industrie, la courbe "Industry avg" (volatilité moyenne de toute l'industrie) se retrouve **au-dessus** des courbes positive et negative, au lieu d'être entre les deux.

**Explication** — biais de sélection :
- Les courbes positive/negative moyennent les tickers **qui ont une CEO letter avec un sentiment détecté** — un sous-ensemble de l'industrie.
- La courbe "Industry avg" est calculée sur **tous les tickers** de l'industrie, y compris ceux sans CEO letter.
- Les sociétés qui publient une lettre CEO structurée et parseable sont généralement les plus grandes / les plus établies → **moins volatiles**. Les small caps illiquides (GEVO, LOOP, SNES…) tirent la moyenne industrie vers le haut.

En résumé : `positive ∪ negative ≠ industrie entière`, et le complément (tickers non labellisés) est systématiquement plus volatil.

---

## 7. Inversion positive/negative entre raw et unbiased volatilité

**Observation** : sur le raw volatility, **negative > positive**. Mais sur le unbiased volatility, **positive > negative** — l'ordre s'inverse.

**Explication** : la soustraction ne porte pas sur la même baseline pour les deux groupes. Chaque (ticker, year) a une date de publication différente, et l'industrie est échantillonnée à **des dates calendaires différentes** :

- **Negative** : publications pendant des périodes où **toute l'industrie** est déjà volatile (crises, stress de marché). Donc `raw_vol(neg)` est élevée, mais `industry_vol(aux dates neg)` l'est **aussi** — la soustraction comprime la valeur unbiased.
- **Positive** : publications pendant des périodes plus calmes. `raw_vol(pos)` est plus basse, mais `industry_vol(aux dates pos)` est encore plus basse — la valeur unbiased reste relativement haute.

**Implication** : la sur-volatilité raw des sentiments négatifs est entièrement **expliquée par les conditions de marché** au moment de la publication, pas par le sentiment lui-même. L'unbiased corrige cet effet de confusion et révèle un signal différent.

---

## 8. Ajout de métriques : pondération volume-weighted et normalisation à J0

### Pondération volume-weighted de l'industrie

La sectorisation initiale utilisait une moyenne **équipondérée**. On a ajouté une pondération par le volume moyen de chaque ticker :

$$
w_j = \frac{\bar{V}_j}{\sum_{k=1}^{N} \bar{V}_k}
$$

$$
\bar{r}^{\text{vw}}_{\text{industrie}, t} = \sum_{j=1}^{N} w_j \, r_{j,t}
$$

Cela donne des métriques "unbiased VW" :

$$
r^{\text{adj,vw}}_{i,t} = r_{i,t} - \bar{r}^{\text{vw}}_{\text{industrie}, t}
$$

et de même pour le volume (`volume_unbiased_vw`) et le rendement cumulé (`cum_return_unbiased_vw`). L'idée est que les gros tickers (plus liquides) reflètent mieux le "vrai" comportement sectoriel.

### Normalisation par rapport à J0

Pour rendre les event studies comparables en dynamique plutôt qu'en niveau, on normalise chaque métrique par sa valeur au jour de publication :

$$
m^{\text{norm}}_{i,d} = \frac{m_{i,d} - m_{i,0}}{|m_{i,0}|}
$$

Cela visualise la **variation relative** de chaque grandeur par rapport à son niveau le jour de la publication. On obtient des versions `norm_*` de toutes les métriques (volatilité, volume, en brut et non biaisé).

**Pourquoi des plots normalisés ?** Les métriques brutes (volatilité, volume) ont des niveaux absolus très différents d'un ticker à l'autre et d'une industrie à l'autre. En normalisant à J0, on répond à la question : *"comment la métrique évolue-t-elle relativement à son niveau au moment exact de la publication ?"* plutôt que *"quel est le niveau moyen ?"*. Cela permet de comparer des dynamiques (montée, creux, plateau) entre sentiments et entre industries sans que les différences d'échelle ne masquent les patterns.

**Attention** : cette normalisation explose quand $m_{i,0} \approx 0$ (typiquement `volume_unbiased_vw` qui peut être quasi-nul). Un ylim basé sur les percentiles 2%–98% est appliqué sur les plots pour éviter que quelques outliers ne rendent le graphe illisible.

---

## 9. Tableau récapitulatif de tous les indicateurs

### Métriques brutes (event studies + bar charts)

| Nom dans le code | Formule | Motivation |
|---|---|---|
| `raw_volatility` | $\sigma_{i,t} = \text{std}_{20}(r_{i,\cdot})$ | Volatilité intrinsèque du titre — signal de base |
| `raw_volume` | $V^{\text{ATS}}_{i,t} = V_{i,t} / \bar{V}_i$ | Activité de trading normalisée par l'historique du titre |
| `raw_price_norm` | $\tilde{P}_{i,d} = P_{i,t_0+d} / P_{i,t_0}$ | Prix normalisé à J0=1 pour rendre les tickers comparables |
| `unbiased_volatility` | $\sigma_{i,t} - \bar{\sigma}_{\text{ind}, t}$ | Volatilité propre au titre, nette de l'effet sectoriel |
| `unbiased_volume` | $V^{\text{ATS}}_{i,t} - \bar{V}^{\text{ATS}}_{\text{ind}, t}$ | Volume propre au titre, net de l'effet sectoriel (EW) |
| `volume_unbiased_vw` | $V^{\text{ATS}}_{i,t} - \bar{V}^{\text{ATS,vw}}_{\text{ind}, t}$ | Volume propre, net de l'industrie pondérée par volume — moins bruité par les small caps |
| `cum_return` | $\prod_{k=1}^{d}(1+r_{i,t_0+k}) - 1$ | Rendement cumulé depuis J0 — performance post-publication |
| `cum_return_unbiased` | `cum_return` $-$ `cum_return_industrie_EW` | Surperformance vs industrie (équipondérée) depuis J0 |
| `cum_return_unbiased_vw` | `cum_return` $-$ `cum_return_industrie_VW` | Surperformance vs industrie (volume-weighted) depuis J0 |

### Métriques normalisées à J0 (event studies uniquement)

| Nom dans le code | Formule | Motivation |
|---|---|---|
| `norm_unbiased_volatility` | $(m_d - m_0) / |m_0|$ sur `unbiased_volatility` | Dynamique relative de la vol non biaisée — compare la forme des courbes entre sentiments |
| `norm_unbiased_volume` | $(m_d - m_0) / |m_0|$ sur `unbiased_volume` | Idem pour le volume non biaisé EW |
| `norm_raw_volatility` | $(m_d - m_0) / |m_0|$ sur `raw_volatility` | Dynamique relative de la volatilité brute |
| `norm_raw_volume` | $(m_d - m_0) / |m_0|$ sur `raw_volume` | Dynamique relative du volume brut |
| `norm_volume_unbiased_vw` | $(m_d - m_0) / |m_0|$ sur `volume_unbiased_vw` | Dynamique relative du volume non biaisé VW |

### Métriques agrégées (bar charts uniquement, fenêtre de 5 jours)

| Nom dans le code | Formule | Motivation |
|---|---|---|
| `cum_return_1d` | Rendement cumulé J0→J+1 | Réaction immédiate du marché |
| `cum_return_5d` | Rendement cumulé J0→J+5 | Réaction à une semaine |
| `cum_return_1d_unbiased` | `cum_return_1d` $-$ industrie EW | Alpha court terme vs industrie |
| `cum_return_5d_unbiased` | `cum_return_5d` $-$ industrie EW | Alpha moyen terme vs industrie |
| `cum_return_1d_unbiased_vw` | `cum_return_1d` $-$ industrie VW | Alpha court terme vs industrie VW |
| `cum_return_5d_unbiased_vw` | `cum_return_5d` $-$ industrie VW | Alpha moyen terme vs industrie VW |

### Légende des notations

- $\bar{\cdot}_{\text{ind}, t}$ : moyenne industrie au jour $t$ (équipondérée sauf mention VW)
- $\bar{\cdot}^{\text{vw}}_{\text{ind}, t}$ : moyenne industrie pondérée par le volume moyen de chaque ticker
- EW = Equal-Weighted, VW = Volume-Weighted
- J0 = jour de publication du rapport annuel (10-K)
- $d \in [-10, +10]$ : jours de trading relatifs à J0

---

## 10. Prochaine étape : jugement LLM sur les KPIs extraits

L'étape suivante consiste à demander au LLM de juger la qualité / pertinence des KPIs extraits par OCR (revenue, net income, etc.) pour chaque rapport annuel. Le LLM doit évaluer si les valeurs extraites sont correctes, cohérentes et exploitables.

Une fois ce label de qualité obtenu par document, on refait exactement la même analyse d'event study (mêmes indicateurs : volatilité, volume, rendements cumulés, métriques unbiased/VW/normalisées) mais en conditionnant cette fois sur le jugement KPI du LLM (bon / mauvais / ambigu) au lieu du sentiment CEO letter. Cela permettra de tester si la qualité des résultats financiers (tels que perçus par le LLM à partir des données OCR) génère un signal de marché plus net que le ton de la lettre CEO.

---

## 11. Analyse par stock individuel : effet des publications sur les indicateurs

On cherche à vérifier, pour un stock donné, si les publications des rapports annuels ont effectivement un effet sur les indicateurs de marché (returns, volatilité, volume). Pour chaque ticker, on plot un event study avec **une courbe par fiscal year** (FY2017–FY2022) ainsi qu'une courbe moyenne en gras. Cela permet de visualiser la variabilité inter-année et d'identifier si certains rapports génèrent un signal plus fort que d'autres.

### Stocks sélectionnés

| Ticker | Industrie |
|--------|-----------|
| BCPC | Specialty Chemicals |
| GEVO | Specialty Chemicals |
| AZO | Auto Parts |
| ORLY | Auto Parts |
| SLB | Oil & Gas Equipment & Services |
| CPB | Packaged Foods |
| AGNC | REIT Mortgage |
| HWKN | Specialty Chemicals |
| CLMT | Specialty Chemicals |
| LOOP | Specialty Chemicals |

Script : `KPI_analysis/event_single_stock.py`. Plots dans `output/plots/sentiment_summary/{TICKER}/` (±10 jours) et `output/plots/sentiment_summary_augmented_lag/{TICKER}/` (±90 jours).

---

## 12. Extension de la fenêtre d'étude : ±90 jours

On étend la fenêtre d'observation de ±10 à ±90 jours de trading autour de la date de publication. Pour limiter le nombre de points sans perdre d'information loin de J0, on échantillonne :

- Tous les 10 jours de J-90 à J-20 : $\{-90, -80, -70, -60, -50, -40, -30, -20\}$
- Quotidien de J-10 à J+10 : $\{-10, -9, \ldots, +9, +10\}$
- Tous les 10 jours de J+20 à J+90 : $\{+20, +30, +40, +50, +60, +70, +80, +90\}$

Soit 37 points au total, espacés uniformément sur l'axe x des graphes.

### Correction de la formule des rendements cumulés

Une erreur a été corrigée dans le calcul des `return_t{d}`. La formule correcte, ancrée à J0 :

$$
\text{return\_t}\{d\} = \frac{P_{t_0 + d} - P_{t_0}}{P_{t_0}}
$$

pour tout $d$ (négatif ou positif). Concrètement :

- Si $d > 0$ : on regarde combien le prix a monté/baissé **après** J0 → $(P_{t_0+d} - P_{t_0}) / P_{t_0}$
- Si $d < 0$ : on regarde où était le prix **avant** J0 par rapport à J0 → $(P_{t_0+d} - P_{t_0}) / P_{t_0}$

Ainsi `return_t{0} = 0` par construction, et la courbe donne le rendement cumulé vu depuis le jour de publication dans les deux directions.

---

## 13. Distributions split par le signe du return à J-90

On cherche à caractériser les publications selon l'historique récent du titre. Pour chaque (ticker, year), on calcule `return_t{-90}` (rendement cumulé des 90 jours **avant** la publication) et on sépare en deux groupes :

- **Vert** : le titre a monté dans les 90 jours précédant la publication (`return_t{-90} > 0`)
- **Rouge** : le titre a baissé dans les 90 jours précédant la publication (`return_t{-90} < 0`)

On superpose les histogrammes des deux groupes pour chaque métrique : returns à différents horizons (J+1, J+5, J+10, J+90), volatilité et volume au jour de publication. Le 0 est forcé comme bord de bin pour les métriques de return afin que la séparation visuelle soit nette.

Script : `KPI_analysis/distributions_by_return_m90.py`. Plots dans `output/plots/distributions_by_return_m90/`.

---

## 14. Distribution des différences J+1 vs J-1 (effet immédiat de la publication)

On cherche à mesurer l'**effet immédiat** de la publication du rapport annuel en regardant, pour chaque (ticker, year), la différence entre la valeur de l'indicateur le lendemain vs la veille de la publication :

- $\Delta \text{vol} = \text{Volatility}(t_0+1) - \text{Volatility}(t_0-1)$
- $\Delta \text{volume} = \text{Volume\_ATS}(t_0+1) - \text{Volume\_ATS}(t_0-1)$
- $\Delta \text{return} = \text{return\_t}\{+1\} - \text{return\_t}\{-1\}$

On étudie ensuite la distribution de ces trois différences sur l'ensemble des rapports pour voir si la publication génère un shift systématique (volume en hausse, volatilité en hausse, etc.) ou si l'effet est noyé dans le bruit.

---

## 15. Earnings date vs 10-K filing : où est la vraie réaction de marché ?

### Distinction fondamentale

Pour une entreprise US cotée, la publication des résultats annuels se fait en **deux temps** :

1. **Earnings call** (= earnings date, J0) : l'entreprise publie un **press release** avec les chiffres clés (revenue, EPS, marges) + tient une conférence téléphonique avec les analystes + donne la **guidance** (prévisions pour les prochains trimestres). C'est la **première fois** que l'information est rendue publique → le marché réagit immédiatement (gros volume, gros move de prix).

2. **10-K filing** (J+20 à J+40) : dépôt sur EDGAR du document réglementaire complet (~200 pages : notes comptables, risques, litiges, rapport d'audit). Les headline numbers sont identiques à ceux du press release → comme il n'y a pas de surprise, le marché ne réagit quasiment pas.

### EPS et Surprise

L'**EPS** (Earnings Per Share) mesure le bénéfice par action :

$$
\text{EPS} = \frac{\text{Net Income} - \text{Preferred Dividends}}{\text{Weighted Average Shares Outstanding}}
$$

Où :

- **Net Income** (résultat net) : bénéfice final après déduction de tous les coûts, taxes, intérêts, dépréciations. C'est le dernier chiffre du compte de résultat ("bottom line").
- **Preferred Dividends** (dividendes préférentiels) : dividendes fixes versés en priorité aux actionnaires préférentiels (pas de droit de vote, mais dividende garanti). On les soustrait car l'EPS mesure ce qui revient aux actionnaires **ordinaires**. En pratique, beaucoup de sociétés n'ont pas d'actions préférentielles, donc ce terme est souvent 0.
- **Weighted Average Shares Outstanding** (nombre moyen pondéré d'actions) : moyenne du nombre d'actions en circulation, pondérée par le temps. Le nombre d'actions change en cours d'année (rachats, émissions, splits), donc on pondère. Ex : 100M d'actions de jan–juin, 90M de juil–déc → WASO = $100 \times \frac{6}{12} + 90 \times \frac{6}{12} = 95$M.

Il existe deux variantes :
- **Basic EPS** : nombre d'actions effectivement en circulation
- **Diluted EPS** : inclut les actions potentielles (stock-options, obligations convertibles, warrants). C'est la version conservatrice, suivie par les analystes pour le consensus.

L'**EPS Estimate** = le consensus des analystes **sell-side** (Goldman, JPM, Morgan Stanley, etc.) compilé par un agrégateur de données (**Refinitiv/LSEG** dans le cas de Yahoo Finance ; Bloomberg, FactSet pour d'autres plateformes). Ce ne sont ni l'entreprise ni des gens mandatés par elle — ce sont des analystes indépendants qui publient leurs propres modèles de valorisation. L'entreprise peut donner du "guidance" (ses propres prévisions), mais le consensus est la moyenne des estimations externes. Le nombre d'analystes varie : une large-cap peut avoir 20+ analystes, une small-cap 2-3.

C'est généralement l'**EPS ajusté (non-GAAP)** qui est suivi — l'entreprise et les analystes s'accordent implicitement sur quelles charges exceptionnelles exclure. L'EPS GAAP du 10-K peut différer sensiblement.

La **Surprise** mesure l'écart entre réalisé et attendu :

$$
\text{Surprise (\%)} = \frac{\text{Reported EPS} - \text{EPS Estimate}}{|\text{EPS Estimate}|} \times 100
$$

C'est la variable qui fait bouger le cours : pas le niveau absolu des résultats, mais l'**écart vs les attentes**. Exemple : EPS attendu 0.50\$, annoncé 0.56\$ → surprise = +12% → le cours monte.

La **guidance** (prévisions du management) peut être aussi importante que les résultats eux-mêmes : une boîte peut battre le consensus EPS mais chuter si elle abaisse ses prévisions annuelles.

### Réaction aux 4 trimestres

Le marché réagit à **chaque** earnings call trimestriel (Q1, Q2, Q3, Q4), pas uniquement au Q4. Dans notre étude on se concentre sur le Q4 car c'est celui qui correspond au rapport annuel / 10-K / CEO letter analysé par OCR. Mais les Q1–Q3 génèrent aussi des réactions (parfois plus fortes car ils signalent la tendance en cours d'année).

### Implication pour notre event study

Si l'event study ancrée sur la date de **10-K filing** ne montre pas de signal clair (vol/volume/return), c'est normal : l'information a déjà été absorbée 3-6 semaines plus tôt lors de l'earnings call. Pour tester la vraie réaction de marché aux résultats annuels, il faut ancrer l'étude sur l'**earnings date**.

Script : `KPI_analysis/event_study_earnings.py`. La date d'earnings est récupérée via `yfinance.get_earnings_dates()` et mappée au fiscal year en cherchant l'earnings call dans la fenêtre [filing - 60j, filing - 5j] avant le 10-K.

---

## 16. Bug "Industry avg" volatilité REIT-Mortgage explosée (~0.25-0.40)

La courbe "Industry avg" du plot `sentiment_vs_indicators.py` pour Real Estate / REIT-Mortgage affichait une volatilité de 0.25-0.40 alors que les stocks individuels étaient à ~0.015. **Cause : cache périmé** (`cache/industry_indicators/Real_Estate___REIT_-_Mortgage.csv`). Le cache avait été généré avec d'anciennes données yfinance contenant un ou plusieurs tickers avec des prix aberrants (penny stock / delisted). Après suppression et regénération du cache, la valeur retombe à ~0.018 (cohérent avec les stocks individuels). Toujours vérifier la date du cache si les valeurs industry semblent incohérentes.

---

## 17. Deux façons de calculer la "moyenne industrie" dans les event studies

Les event studies utilisent deux courbes d'apparence similaire mais de construction très différente :

### Méthode 1 — "Industry Avg Volatility" (moyenne transversale à date fixe)

Pour chaque event $(i, \text{year})$ au jour relatif $d$ :
1. On lit `industry_df["volatility"]` à la date calendaire $t_0 + d$ — c'est la volatilité moyenne **de tous les peers** de l'industrie ce jour-là (32 à 53 tickers selon l'industrie), calculée indépendamment de qui a ou n'a pas un filing ce mois-là.
2. On moyenne ensuite sur tous les events de l'industrie.

$$
\text{IndustryAvg}(d) = \frac{1}{|\mathcal{E}|} \sum_{e \in \mathcal{E}} \bar{\sigma}_{\text{ind}}(t_0^{(e)} + d)
$$

où $\bar{\sigma}_{\text{ind}}(t)$ est la moyenne équipondérée de la volatilité rolling-20d de **tous** les tickers de l'industrie au jour $t$.

**Caractéristiques** : inclut systématiquement tous les peers (y compris ceux sans filing, small caps illiquides, etc.). C'est un **benchmark passif** — "que fait l'industrie en général à ces dates ?"
C'était la méthode utilisait pour calculer les avg industry dans sentiment_vs_indicators. Comme c'était lissé on voyait pas de réaction du marché au global.


### Méthode 2 — "Stock Volatility" (moyenne des stocks ayant un event)

Pour chaque event $(i, \text{year})$ au jour relatif $d$ :
1. On lit `prices["Volatility"]` du stock **individuel** $i$ qui a déposé son 10-K ce jour-là (sa propre volatilité rolling-20d).
2. On moyenne sur les events de la même industrie.

$$
\text{StockVol}(d) = \frac{1}{|\mathcal{E}|} \sum_{e \in \mathcal{E}} \sigma_{i(e)}(t_0^{(e)} + d)
$$

où $\sigma_{i(e)}(t)$ est la volatilité rolling-20d du ticker $i$ de l'event $e$.

**Caractéristiques** : ne porte que sur le **sous-ensemble** de tickers qui ont effectivement un event (filing + CEO letter parseable). Ce sous-ensemble est biaisé vers les large caps bien couvertes → typiquement **moins volatiles** que la moyenne industrie (cf. §6).

### Pourquoi les deux courbes diffèrent

| | Industry Avg | Stock Volatility |
|---|---|---|
| **Population** | Tous les peers ($N$ = 32–53) | Seulement les tickers avec un event |
| **Biais** | Inclut les small caps volatiles | Biaisé large-cap (moins volatile) |
| **Niveau typique** | Plus élevé | Plus bas |
| **Ce qu'on mesure** | Contexte de marché sectoriel | Comportement propre des stocks étudiés |

L'écart entre les deux (Industry Avg > Stock Vol) confirme le biais de sélection décrit en §6 : les sociétés dont on analyse la CEO letter sont systématiquement moins volatiles que l'ensemble de l'industrie.

---

## 18. Différence de taille d'échantillon entre les scripts (n=1234 vs n=1091 vs n=1000)

Les graphes produits par `distribution_all_stocks.py` et `event_study_earnings.py` n'ont pas le même nombre d'events malgré la même source (`companies.json`, 244 tickers × 6 ans). L'écart vient des filtres successifs :

| Échantillon | Script | Critère |
|---|---|---|
| **n=1234** | `distribution_all_stocks.py` | Tout (ticker, year) ayant un filing date EDGAR + prix valides ±10j |
| **n=1091** | `event_study_earnings.py` (courbe filing) | Sous-ensemble de n=1234 dont le ticker a des earnings dates sur yfinance (élimine ~143 events : LSE/AIM, small caps OTC sans earnings calendar) |
| **n=1000** | `event_study_earnings.py` (courbe earnings) | Sous-ensemble de n=1091 où `find_q4_earnings_date()` a trouvé un earnings call Q4 dans la fenêtre [filing−60j, filing+1j] (élimine ~91 events sans match Q4) |

**Pourquoi n=1091 < n=1234** : `event_study_earnings.py` commence par chercher les earnings dates via `yfinance.get_earnings_dates()`. Les tickers sans aucune donnée earnings (LSE, AIM, certaines small caps OTC) sont éliminés dès cette étape, avant même de traiter le filing date.

**Pourquoi n=1000 < n=1091** : parmi les tickers avec un earnings calendar, certains n'ont pas de Q4 earnings call dans la fenêtre temporelle attendue (ex : fiscal year exotique, données manquantes sur yfinance pour les années anciennes).

Les deux graphes (Stock Volatility by Industry vs Earnings/Filing overlay) sont néanmoins cohérents : la courbe noire "Weighted avg all" du premier (n=1234) est à ~0.030 à J0, ce qui coïncide avec la courbe filing bleue du second (n=1091, ~0.029 à J0). Les 143 events supplémentaires (LSE/AIM) ne changent pas significativement la moyenne.

### Détail du matching earnings par industrie

| Match | Industrie | US stocks | US rapports | Non-US stocks | Non-US rapports | Total rapports |
|---|---|---|---|---|---|---|
| Matched | Specialty Chemicals | 42 | 234 | 0 | 0 | 234 |
| Matched | Auto Parts | 33 | 163 | 0 | 0 | 163 |
| Matched | Packaged Foods | 32 | 158 | 0 | 0 | 158 |
| Matched | Oil & Gas E&P | 37 | 158 | 0 | 0 | 158 |
| Matched | Oil & Gas Equipment & Services | 28 | 137 | 0 | 0 | 137 |
| Matched | REIT - Mortgage | 31 | 163 | 0 | 0 | 163 |
| Unmatched | Specialty Chemicals | 13 | 54 | 1 | 6 | 60 |
| Unmatched | Auto Parts | 13 | 59 | 1 | 6 | 65 |
| Unmatched | Packaged Foods | 17 | 64 | 0 | 0 | 64 |
| Unmatched | Oil & Gas E&P | 30 | 136 | 4 | 24 | 160 |
| Unmatched | Oil & Gas Equipment & Services | 20 | 79 | 0 | 0 | 79 |
| Unmatched | REIT - Mortgage | 8 | 29 | 0 | 0 | 29 |
| **Total** | — | **238** | **1434** | **6** | **36** | **1470** |

Observations :
- Tous les matched sont US (yfinance earnings calendar disponible uniquement pour les tickers US).
- Les 6 tickers non-US (LSE : ELM.L, ABDP.L, GKP.L, ENQ.L, GENL.L, PHAR.L) sont systématiquement unmatched.
- Parmi les unmatched US (101 tickers, 421 rapports), la majorité sont des small caps dont yfinance n'a pas d'earnings calendar historique avant 2020.

