# Vehicle Test Data Analyzer — Automotive Data Analytics & Anomaly Detection

Pipeline de traitement et d'analyse de données issues de **tests automobiles** (télémétrie 1 Hz : vitesse, régime, température moteur, pression d'huile, débit carburant, batterie, vibrations), avec **détection d'anomalies par Isolation Forest** et un **dashboard interactif Streamlit / Plotly**.

**Stack :** Python · Pandas · PySpark · Scikit-learn · Streamlit · Plotly · SQL (SQLite) · Docker

> Les données sont **synthétiques** (générées par `src/vta/data_generator.py`) avec des défauts injectés à des endroits connus, ce qui permet de *mesurer* la qualité de la détection. Pour des données réelles, il suffit de fournir les mêmes CSV dans `data/raw/` (voir [Données](#données)).

---

## Architecture 

```
                 ┌────────────────────────── pipeline (Pandas ou PySpark) ──────────────────────────┐
 data/raw/*.csv  │  dédoublonnage → tri → valeurs hors plage = NULL → imputation courte (≤ 5 s)       │
 (mesures, essais│  → features (accélération, pente de température, écart-type du régime, vibrations  │
  véhicules)  ──►│     lissées) → KPIs par essai → rapport de qualité des données                    │
                 └───────────────────────────────────────┬────────────────────────────────────────────┘
                                                         ▼
                       ┌──────────── détection d'anomalies (scikit-learn) ────────────┐
                       │ modèles de contexte (HistGradientBoosting) → résidus par capteur │
                       │ → Isolation Forest par type d'essai → événements + sévérité     │
                       │ → capteur suspect (attribution robuste médiane/MAD)             │
                       └───────────────────────────────┬──────────────────────────────┘
                                                       ▼
                       SQLite (schéma relationnel + vues + requêtes analytiques à fenêtres)
                                                       ▼
                       Dashboard Streamlit / Plotly  (5 onglets, détection recalculable en direct)
```

## Structure du dépôt

```
├── src/vta/
│   ├── config.py            # chemins, plages de validité des capteurs, features, paramètres du modèle
│   ├── data_generator.py    # génération de données d'essais + injection de défauts (vérité terrain)
│   ├── pipeline_pandas.py   # nettoyage, features, KPIs, qualité (Pandas)
│   ├── pipeline_spark.py    # même logique en PySpark (fonctions de fenêtre) → Parquet
│   ├── anomaly.py           # résidus contextuels + Isolation Forest + événements + évaluation
│   ├── storage.py           # SQLite : schéma, écriture, lecture seule, requêtes nommées
│   ├── run_pipeline.py      # CLI de bout en bout
│   └── sql_report.py        # exécute sql/queries.sql et affiche les résultats
├── sql/schema.sql           # tables, index, clés étrangères, vue v_test_summary
├── sql/queries.sql          # 7 requêtes analytiques (RANK, AVG/MAX OVER, CTE…)
├── app/streamlit_app.py     # dashboard
├── tests/                   # 23 tests (pytest)
├── Dockerfile, docker-compose.yml, Makefile, requirements.txt
└── docs/PITCH.md            # lignes de CV + questions d'entretien préparées
```

## Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # PySpark nécessite Java 17+ pour le moteur Spark

make pipeline        # génère les données, nettoie, détecte, écrit data/vehicle_tests.db (~10 s)
make dashboard       # http://localhost:8501
make test            # 23 tests
make pipeline-spark  # même pipeline avec PySpark (~30 s, démarrage de la JVM compris)
make sql             # affiche les rapports SQL
```

Sans `make` : `PYTHONPATH=src python -m vta.run_pipeline --engine pandas --regenerate`.
Si tu lances le dashboard sans base de données, un bouton permet de tout générer en un clic.

**Docker** (pipeline Spark puis dashboard) :

```bash
docker compose up --build        # → http://localhost:8501
docker compose --profile test run --rm tests
```

Options utiles de la CLI : `--engine {pandas,spark}`, `--contamination 0.03`, `--n-estimators 400`, `--regenerate`, `--n-vehicles 20`, `--seed 1`.

## Données

12 véhicules (essence/diesel), 4 essais chacun parmi `urban`, `highway`, `mixed`, `acceleration`, `cold_start` : **48 essais, ≈ 56 600 mesures, 15,7 h** de données à 1 Hz.

Le générateur modélise des couplages réalistes (vitesse → rapport → régime ; charge → température avec montée en température à froid ; régime → pression d'huile ; etc.) et injecte :

| Défaut injecté | Capteur | Catégorie |
|---|---|---|
| surchauffe (rampe jusqu'à +22…34 °C) | température moteur | comportement |
| chute de pression d'huile (−40…60 %) | pression d'huile | comportement |
| rafales de vibrations (×3…5) | vibrations | comportement |
| sous-tension batterie (11…12,2 V) | batterie | comportement |
| surconsommation (×2,2…3,2) | débit carburant | comportement |
| pic de régime sans changement de vitesse | régime | comportement |
| trous de données / valeur sentinelle `-999` | aléatoire | qualité (traité par le nettoyage) |

Le fichier brut est volontairement sale : lignes dupliquées, ordre mélangé, valeurs manquantes, sentinelles. Deux véhicules sont « usés » (consommation +18 %, vibrations +50 %) : c'est visible dans les graphiques de comparaison.

**Brancher tes propres données** : fournir dans `data/raw/` `measurements.csv` (`test_id, timestamp, speed_kmh, rpm, throttle_pct, engine_temp_c, oil_pressure_bar, fuel_rate_lph, battery_v, vibration_g`), `tests.csv` et `vehicles.csv` (voir `sql/schema.sql` pour les colonnes). `anomaly_labels.csv` est optionnel (évaluation).

## Pipeline de traitement

1. **Dédoublonnage** sur `(test_id, timestamp)` et tri chronologique.
2. **Valeurs hors plage → NULL** (plages physiques dans `config.py`) : élimine les sentinelles `-999` et les erreurs capteur.
3. **Imputation courte** : report de la dernière valeur valide pour les trous ≤ 5 s ; au-delà, la ligne est marquée `is_valid = false` et n'est pas scorée (pas d'invention de données).
4. **Features** : accélération, pente de température sur 10 s, écart-type glissant du régime, moyenne glissante des vibrations, consommation instantanée.
5. **KPIs par essai** : distance, vitesse moy./max, consommation L/100 km, température max, % de ralenti, freinages brusques (< −3 m/s²), pression d'huile et tension batterie minimales.
6. **Rapport qualité** par essai : doublons supprimés, valeurs hors plage, manquantes avant/après imputation.

**Pandas vs PySpark.** `pipeline_pandas.py` et `pipeline_spark.py` implémentent *exactement* les mêmes transformations (le report de valeur est une fenêtre `last(ignorenulls)` en Spark). `tests/test_spark_equivalence.py` vérifie qu'ils produisent les mêmes features, KPIs et métriques qualité, et une exécution complète donne des détections identiques (1 136 secondes signalées, 90 événements dans les deux cas). Sur 56 k lignes Pandas est plus rapide (< 1 s contre ≈ 18 s, surtout du démarrage JVM) : Spark n'a d'intérêt qu'à l'échelle de millions de lignes ; la version Spark est là pour montrer que la logique est portable sans réécriture conceptuelle.

## Détection d'anomalies

**Pourquoi pas un Isolation Forest directement sur les capteurs ?** Une valeur brute est ambiguë : 8 L/h de carburant est normal à plein régime et anormal au ralenti. Les défauts « noyés » dans la plage normale d'un capteur passent inaperçus.

**Approche retenue** (`src/vta/anomaly.py`) :

1. Pour 5 capteurs (huile, carburant, vibrations, batterie, régime), un petit modèle `HistGradientBoostingRegressor` prédit la valeur attendue à partir du contexte de fonctionnement (ex. pression d'huile ~ régime, température). On garde le **résidu** `mesuré − attendu`.
2. Un **Isolation Forest par type d'essai** (le réchauffage à froid serait une anomalie sur autoroute) est entraîné sur les résidus + température, pente de température, accélération. Les arbres étant invariants d'échelle, aucune normalisation n'est nécessaire.
3. Les secondes signalées sont fusionnées en **événements** (trou ≤ 3 s, durée ≥ 3 s) avec une **sévérité** (low/medium/high) selon la marge au-dessus du seuil de décision.
4. **Attribution** : pour chaque point signalé, le capteur dont l'écart robuste (médiane/MAD) est le plus grand est indiqué comme « capteur suspect » (indice de piste, pas un diagnostic causal).

**Résultats mesurés** sur les données synthétiques (76 défauts de comportement injectés, tolérance ±2 s) :

| Configuration | Rappel événements | Précision événements |
|---|---|---|
| Capteurs bruts → Isolation Forest (contamination 2 %) | 51 % | 39 % |
| **Résidus contextuels → Isolation Forest (2 %, défaut)** | **71 %** | **81 %** |
| idem, contamination 1 % | 50 % | 88 % |
| idem, contamination 3 % | 83 % | 77 % |

Taux de détection par défaut (contamination 2 %) : surchauffe 100 %, pic de régime 92 %, vibrations 67 %, batterie 64 %, surconsommation 60 %, chute de pression d'huile 45 %.

La **contamination** est le compromis rappel/précision : c'est un réglage métier (combien de fausses alertes l'équipe tolère-t-elle ?). Le dashboard permet de la faire varier en direct et d'observer l'effet sur les métriques.

Un test de non-régression (`test_quality_gate_on_generated_fleet`) impose rappel et précision ≥ 60 % et exige que les résidus battent les capteurs bruts d'au moins 15 points de rappel.

## Dashboard

| Onglet | Contenu |
|---|---|
| **Vue d'ensemble** | KPIs flotte, consommation et taux d'anomalies par véhicule, dérive de vibrations, consommation vs vitesse, qualité des données |
| **Analyse d'un essai** | séries temporelles multi-capteurs, points et zones anormaux surlignés, KPIs de l'essai, table des événements |
| **Comparer des essais** | superposition d'un capteur (lissage réglable) sur 2–4 essais, KPIs côte à côte, radar normalisé |
| **Détection d'anomalies** | Isolation Forest recalculé en direct (contamination, nb d'arbres, durée min), métriques vs vérité terrain, scores, capteurs suspects, export CSV |
| **SQL** | requêtes prédéfinies + zone libre en lecture seule (un seul `SELECT`/`WITH`, connexion SQLite `mode=ro`) |

## SQL

`sql/schema.sql` définit un modèle relationnel (`vehicles` 1—n `tests` 1—n `measurements`, `test_kpis`, `data_quality`, `anomaly_events`, table de vérité terrain) avec clés étrangères, index et la vue `v_test_summary`. `sql/queries.sql` contient 7 requêtes analytiques commentées : classement de consommation par type de carburant (`RANK() OVER`), écart à la moyenne du type d'essai (`AVG() OVER`), maximum glissant sur 1 minute autour du pic de température (cadre de fenêtre + CTE), etc. `make sql` les exécute.

## Tests

`pytest` : 23 tests — nettoyage (doublons, sentinelles, trous courts/longs), features et KPIs sur des cas calculables à la main, détection sur outliers synthétiques, fusion d'événements, évaluation, persistance du modèle, intégrité SQLite (clés étrangères, cohérence KPI/événements, requêtes), garde SQL, équivalence Pandas/PySpark, et seuil de qualité de détection.

## Limites et pistes d'amélioration

- **Données synthétiques** : les métriques prouvent que la méthode fonctionne dans un cadre contrôlé, pas qu'elle se généralise à de vraies données ; l'évaluation sur données réelles demanderait des annotations d'ingénieurs.
- **Entraînement et scoring sur le même lot** (analyse hors-ligne). Pour un usage continu : entraîner sur des essais sains puis scorer les nouveaux (`VehicleAnomalyDetector.save/load` est prêt ; un modèle est écrit dans `data/processed/detector.joblib`).
- L'Isolation Forest est **ponctuel** : pas de modélisation explicite de la dynamique temporelle (voir autoencodeur LSTM, ou des features de fenêtre plus riches).
- Les défauts lents ou « collés » (capteur figé) ne sont pas ciblés ; un test de variance nulle sur fenêtre les détecterait.
- Pistes : suivi de dérive (PSI) entre flottes/périodes, ingestion temps réel (Kafka / Spark Structured Streaming), MLflow pour le suivi d'expériences, PostgreSQL/Parquet partitionné à grande échelle, seuils de sévérité calibrés avec les ingénieurs.
