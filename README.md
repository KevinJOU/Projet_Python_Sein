# Projet_Python_Sein
---
# Prédiction de la Malignité d’une Tumeur du Sein – Webapp

## Objectif de la Webapp :

Cette application web permet de prédire la probabilité qu’une tumeur du sein soit bénigne ou maligne à partir de l'analyse (mesures) des noyaux cellulaires provenant de lames d'anapath numérisées. Ces informations (variables) sont saisies par l’utilisateur (médecin anapath est la cible) et un résultat est donné instantanément.

---

## Choix du Dataset

Les données utilisées proviennent du jeu de données Breast Cancer Wisconsin (Diagnostic) disponible sur Kaggle, contenant 569 observations de tumeurs caractérisées par 30 variables numériques issues d’analyses d’images d'anapth. Les données exploitées étaeint de bonne qualité sans donnée manquante ce qui a facilité le travail de préparation.

Variables principales : rayon, texture, périmètre, surface, douceur, compacité, concavité, points concaves, symétrie, dimension fractale (chacune déclinée en moyenne, erreur-type et valeur maximale).

Variable cible : diagnosis (B = bénin, M = malin).


---

## Choix du Modèle

De part la nature du dataset et de la variable à prédire, plusieurs modèles pouvaient être envisagés allant de la Régression Logistique (le plus connu) au modèle Naive Bayes.
J’ai choisi pour ce projet de tester 2 modèles que je connaissais au moisn de nom, afin de progresser pas à pas dans ce DU à savoir la Régression Logistique justement pour sa simplicité, son interprétabilité et le Radom Forest pour son interprétabilité également que l’on peut visualiser facilement mais également pour sa robustesse.

La webapp s’appuie sur un modèle de **régression logistique**, sélectionné pour sa robustesse, son interprétabilité et ses bonnes performances sur ce type de données. Seules les variables les plus discriminantes du dataset ont été retenues pour simplifier l’interface et garantir la pertinence des prédictions.
J'ai égaalement créé le fichier Python pour une autre app utilisant le modèle Radom Forest par curiosité.

---

## Fonctionnement Global de l’Application

L'applicaiton est très basique :
- L’utilisateur saisit les valeurs de 6 variables cliniques principales (j'ai configuré le pas, le minimum, ainsi que le maximum de chaque variables). 
NB : j’ai potentiellement induit un biais ici car me suis basé sur le maximum de chaque variable de mon dataset. Or on sait qu’ne biologie il peut y avoir des cas particuliers. Pour être plus pertinent, il aurait fallu : soit avoir davantage de cas, soit parler avec des professionnels pour avoir leurs avis et notamment les médecins anapath.

- Les valeurs sont standardisées puis transmises au modèle de régression logistique pré-entraîné.
  
- L’application affiche la prédiction (bénigne ou maligne) ainsi que la probabilité associée, pour chaque cas soumis.


