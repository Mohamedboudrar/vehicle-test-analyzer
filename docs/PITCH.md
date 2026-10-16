# Fiche CV & entretien — Vehicle Test Data Analyzer

> Précise toujours que les données sont **synthétiques** (générées avec défauts injectés pour pouvoir mesurer la détection). C'est une force (évaluation rigoureuse), pas une faiblesse, si tu le dis toi-même.

## Pour le CV (à adapter, 3–4 puces)

**Vehicle Test Data Analyzer — Automotive Data Analytics & Anomaly Detection**
*Python, Pandas, PySpark, Scikit-learn, Streamlit, Plotly, SQL, Docker*

- Conception d'un pipeline Pandas/PySpark (nettoyage, imputation, features, KPIs) traitant 56 k mesures de 48 essais ; équivalence des deux moteurs validée par tests automatisés.
- Détection d'anomalies par Isolation Forest sur résidus contextuels (modèles de référence par capteur) : rappel 71 % / précision 81 % sur défauts injectés, contre 51 % / 39 % avec les capteurs bruts.
- Dashboard Streamlit/Plotly (5 vues : flotte, essai, comparaison, anomalies recalculables en direct, explorateur SQL) sur base SQLite relationnelle avec requêtes à fenêtres.
- Industrialisation : 23 tests pytest, Docker/Compose, CLI reproductible.

## Pitch en 30 secondes

« J'ai construit une chaîne complète d'analyse d'essais véhicules : ingestion de télémétrie brute et sale, nettoyage et features en Pandas et PySpark, détection d'anomalies, stockage SQL et dashboard. Le point technique clé : un Isolation Forest sur les capteurs bruts ne détectait qu'une partie des défauts ; en modélisant d'abord ce qui est *attendu* selon le régime de fonctionnement et en détectant sur les résidus, le rappel passe de 51 % à 71 % et la précision de 39 % à 81 %. Comme les données sont synthétiques avec défauts injectés, j'ai pu mesurer ça proprement. »

## Questions probables

**Comment fonctionne un Isolation Forest ?**
Il construit des arbres qui coupent aléatoirement l'espace (feature aléatoire, seuil aléatoire). Un point anormal est isolé en peu de coupes (chemin court), un point normal en demande beaucoup. Le score est lié à la longueur moyenne du chemin sur la forêt ; la `contamination` fixe le seuil de décision.

**Pourquoi des résidus plutôt que les capteurs bruts ?**
Une valeur n'est anormale que relativement au contexte (8 L/h est normal à plein régime, pas au ralenti). On apprend le comportement attendu avec un modèle supervisé simple et on isole les écarts. Résultat mesuré : +20 points de rappel et +42 points de précision à contamination égale.

**Pourquoi un modèle par type d'essai ?**
Le « normal » dépend du scénario (montée en température à froid ≠ autoroute). Un modèle global confondrait contexte et anomalie.

**Comment choisir la contamination ?**
C'est le curseur rappel/précision, donc une décision métier : coût d'une fausse alerte vs coût d'un défaut raté. Sans labels, on l'ajuste avec les ingénieurs sur un échantillon revu à la main ; avec labels, on trace la courbe rappel/précision (voir le tableau dans le README : 1 % → 50/88, 2 % → 71/81, 3 % → 83/77).

**Comment évaluer sans vérité terrain en réel ?**
Revue d'un échantillon d'événements par des experts (précision estimée), injection de défauts synthétiques sur des essais sains (rappel estimé), stabilité des alertes entre essais comparables, suivi des faux positifs dans le temps.

**Pourquoi PySpark pour 56 k lignes ?**
Honnêtement, Pandas suffit et il est plus rapide ici (le démarrage de la JVM domine). La version Spark montre que la logique (fenêtres partitionnées par essai) passe à l'échelle de millions de lignes sans changer la conception ; j'ai vérifié par test que les deux donnent les mêmes résultats.

**Quel est le risque de fuite / de biais dans ton évaluation ?**
Le modèle est entraîné et évalué sur le même lot (non supervisé, mais transductif), et les données sont synthétiques. En production : entraîner sur des essais sains, scorer des nouveaux essais, surveiller la dérive.

**Que ferais-tu en production ?**
Entraînement périodique sur une référence saine, scoring par lots ou en streaming (Spark Structured Streaming), versionnage du modèle (MLflow), monitoring de dérive (PSI) et des alertes, boucle de retour des ingénieurs pour recalibrer les seuils de sévérité.

**Limites ?**
Détection ponctuelle sans dynamique temporelle explicite ; défauts lents ou capteurs figés non ciblés ; attribution du capteur suspect = indice, pas cause racine ; données synthétiques.
