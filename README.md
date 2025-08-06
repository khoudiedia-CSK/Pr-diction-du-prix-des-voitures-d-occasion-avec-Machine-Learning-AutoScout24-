📊 Prédiction des prix automobiles avec Machine Learning

Contexte du projet:

Ce projet vise à prédire le prix des voitures d’occasion à partir d’un jeu de données riche en caractéristiques techniques et commerciales (marque, modèle, type de carburant, boîte de vitesse, etc.).

L’objectif est d’évaluer la performance de différents modèles de machine learning, principalement Random Forest et XGBoost, pour estimer le prix avec la meilleure précision possible. Une étape clé consiste à optimiser les hyperparamètres via GridSearchCV pour améliorer la qualité des prédictions.

Le jeu de données autoscout.csv contient plusieurs variables explicatives et une cible continue, le prix (price). Le projet suit une démarche classique de data science : exploration, prétraitement, modélisation, optimisation et évaluation.

🧾 Données utilisées
Fichier : autoscout.csv
Colonnes principales :

make_model : marque et modèle

body_type, Fuel, Gearing_Type : caractéristiques du véhicule

price : prix cible à prédire

autres variables techniques et options

🚀 Étapes du projet
1. Chargement et exploration des données
   Chargement du fichier CSV avec pandas

   Analyse des colonnes, valeurs manquantes, types de données

   Visualisation des variables importantes (ex: répartition des prix, types de carburant, marques)


2. Prétraitement et encodage
    Gestion des valeurs manquantes

    Encodage des variables catégorielles avec pd.get_dummies (ex: Fuel, Gearing_Type, make_model)

    Séparation des variables explicatives (features) et cible (price)


3. Séparation des données en train/test
    Utilisation de train_test_split pour créer des jeux d’entraînement et de test

    Test = 20% des données 


5. Modélisation avancée : Random Forest Regressor et XGBoost

    Entraînement du modèle XGBoost sans optimisation

1.Graphique de comparaison des prédictions (Random Forest vs XGBoost)
Ce graphique montre la corrélation entre les prix réels et les prix prédits par les deux modèles. Une droite parfaitement diagonale représenterait une prédiction idéale. On observe que les prédictions de XGBoost sont légèrement plus proches de cette diagonale, indiquant une meilleure qualité de prédiction.
<img width="1043" height="521" alt="image" src="https://github.com/user-attachments/assets/19ea303b-eb5f-4dea-8abd-68ec51abc6f0" />

Résultats:

 Random Forest - MSE: 4791923.65, R²: 0.9112

XGBoost - MSE: 4449720.50, R²: 0.9175

NB :
Même sans optimisation avancée, XGBoost montre une robustesse et une efficacité impressionnantes. 
Ce résultat démontre qu’il constitue une solution solide pour la prédiction des prix de voitures, même en configuration standard.
Évaluation sur test avec MSE et R²
<img width="840" height="487" alt="image" src="https://github.com/user-attachments/assets/540bee53-8284-4f3c-9d30-db6b2594475f" />
<img width="817" height="485" alt="image" src="https://github.com/user-attachments/assets/50d1d51c-6b47-42a6-a06e-9aa41b02f047" />

NB:
Graphiques des erreurs (MSE) et qualité de prédiction (R²) avant optimisation
Ces trois graphiques comparent quantitativement les performances des modèles sans optimisation.
Le MSE plus faible de XGBoost montre qu’il fait moins d’erreurs quadratiques moyennes.
Le R² plus élevé indique que XGBoost explique une plus grande partie de la variance du prix.

6. Optimisation des hyperparamètres avec GridSearchCV
Définition d’une grille d’hyperparamètres (n_estimators, max_depth, learning_rate)

Recherche via validation croisée (3-fold)

Sélection des meilleurs paramètres

Réentraînement du modèle avec ces paramètres

Nouvelle évaluation

Meilleurs paramètres trouvés :
{'learning_rate': 0.2, 'max_depth': 5, 'n_estimators': 150}

Performances améliorées :
MSE: 4,146,879.0
R²: 0.9232

7. Visualisation des résultats

NB: Graphiques comparatifs des performances avant et après GridSearchCV

Le graphique R² montre une augmentation de la capacité explicative du modèle XGBoost après optimisation des hyperparamètres.

Le graphique MSE confirme la réduction de l’erreur quadratique moyenne, signe d’une meilleure précision prédictive.

Ces visualisations illustrent clairement l’intérêt d’un tuning méthodique des hyperparamètres pour maximiser la performance d’un modèle de machine learning.

Graphique comparatif R² avant/après GridSearchCV
<img width="735" height="437" alt="image" src="https://github.com/user-attachments/assets/8053ae5d-48f9-4be0-bfc6-82ebac39f594" />

Graphique comparatif MSE avant/après GridSearchCV

<img width="699" height="468" alt="image" src="https://github.com/user-attachments/assets/92e97d48-3ca5-40b6-b835-956f3653e9a8" />

Résultats principaux:

Modèles testés : Random Forest Regressor et XGBoost.

Métriques utilisées : Mean Squared Error (MSE) et coefficient de détermination (R²).

Performance initiale :
Random Forest : MSE = 4 791 923.65, R² = 0.9112
XGBoost : MSE = 4 449 720.50, R² = 0.9175

Optimisation des hyperparamètres avec GridSearchCV sur XGBoost :
Meilleurs paramètres : learning_rate=0.2, max_depth=5, n_estimators=150
Performance optimisée : MSE = 4 146 879.0, R² = 0.9232

L’optimisation des hyperparamètres améliore significativement la précision, confirmant l’intérêt de ce type d’approche pour les modèles complexes.

📌 Remarques
Le projet montre l’importance de l’optimisation des hyperparamètres pour améliorer la qualité des prédictions

XGBoost avec GridSearchCV donne les meilleurs résultats sur ce dataset

Des améliorations futures peuvent inclure l’analyse plus fine des variables, nettoyage avancé, ou essais avec d’autres modèles
