# OC/DS Projet 3 : Concevez une application au service de la santé publique
Formation OpenClassrooms - Parcours data scientist - Projet Professionnalisant (Octobre-Décembre 2022)

## Secteur : 
Santé 

## Technologies utilisées : 
  * Jupyter Notebook, 
  * Python (pandas, numpy, matplotlib, seaborn, scikit-learn)
  * scikit-learn : kNN, GridSearchCV

## Mots-clés : 
tests statistiques, analyse exploratoire, nettoyage, bag-of-words

## Le contexte : 
L’agence Santé Publique France a lancé un appel à projet autour des problématiques alimentaires. Elle souhaite trouver des idées innovantes d’applications pour améliorer l’alimentation de la population française. 

## La mission : 
Proposer une idée d’application et mener une analyse exploratoire pour vérifier que notre idée est réalisable à partir du jeu de données nutritionnelles mis à notre disposition.

 ## Livrables :
 * notebook_nettoyage.ipynb : notebook du nettoyage des données
 * notebook_exploration.ipynb : notebook d’exploration comportant les analyses uni et multivariées
 * presentation.pdf : support de présentation pour la soutenance

## Méthodologie suivie : 
1. Trouver une idée d’application :
   
«l’utilisateur scanne le code barre du produit, l’app lui propose des produits de la même famille d’aliments, mais avec la meilleure valeur nutritionnelle possible et le plus faible degré de transformation possible»

2. Nettoyer le jeu de données :
* sélectionner les variables pertinentes
* traiter les doublons (garde le produit avec le moins de valeurs manquantes et le plus récemment mis à jour)
* traiter les valeurs aberrantes (supprime, impute avec la moyenne)
* traiter les valeurs manquantes (impute avec 0, avec la moyenne, estime avec kNN)

3. Réaliser une analyse exploratoire :
* analyses univariées (histogrammes)
* analyses multivariées (boxplots, nuages de points, matrice de corrélation)
* tests statistiques (ANOVA, Kruskal-Wallis, chi-2)

## Compétences acquises :  
* Effectuer une analyse statistique multivariée
* Communiquer mes résultats à l’aide de représentations graphiques lisibles et pertinentes
* Effectuer une analyse statistique univariée
* Effectuer des opérations de nettoyage sur des données structurées

## Data source : 
https://world.openfoodfacts.org/

