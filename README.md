# Projet7-OP: Analyse de Sentiment pour Air Paradis

## Description du projet

Ce projet vise à développer un modèle d'intelligence artificielle capable de prédire le sentiment exprimé dans les tweets concernant la compagnie aérienne "Air Paradis". Le projet a pour objectif de créer un prototype fonctionnel sous forme d'API, permettant à "Air Paradis" d'anticiper les bad buzz sur les réseaux sociaux.

## Objectifs

- **Préparer et analyser les données** : Utiliser des données open source pour explorer et préparer les tweets pour la modélisation.
- **Développer plusieurs modèles** : Comparer une approche simple de régression logistique avec des modèles avancés, y compris un modèle basé sur BERT.
- **Déploiement d'une API** : Déployer l'API de prédiction sur une plateforme Cloud en utilisant les principes MLOps pour un déploiement continu.
- **Suivi de la performance** : Utiliser Azure Application Insights pour le suivi des prédictions correctes et incorrectes, et surveiller la performance en production.

## Structure du projet

La structure du projet est organisée pour faciliter le développement, le déploiement et le suivi du modèle d'analyse de sentiment.



```plaintext
projet7-op/ 
├── data/ 			   	     # Dossier pour les jeux de données 
├── notebooks/ 	     	# Dossier pour les notebooks Jupyter 
├── models/ 			       # Dossier pour stocker les modèles entraînés 
├── mlruns/ 		  	     # Dossier pour les enregistrements MLFlow 
├── main.py 		       	# Fichier principal de l'API 
├──templates/ 
 ├── welcome.html 
 ├── predict.html 
 ├── test.html 	     	# Interface de test pour soumettre des tweets 
 ├──adm.html            # Interface d'administration pour sauvegarder et réentraîner 
├── README.md 	        # Fichier de documentation du projet 
├── requirements.txt 	# Fichier pour lister les dépendances Python 
├── .gitignore 	     	# Fichier pour exclure certains fichiers du versioning
```

## Explication de l'API

L'API Flask déployée permet d'analyser les tweets en temps réel et de renvoyer un sentiment prédit (positif ou négatif). Trois interfaces principales sont disponibles :
1. **Page de prédiction (`predict.html`)** : Cette page permet de soumettre des tweets réels pour obtenir une prédiction.
2. **Page de test (`test.html`)** : Interface utilisée pour les tests avec la possibilité de confirmer ou corriger les prédictions.
3. **Page d'administration (`adm.html`)** : Utilisée pour afficher les tweets enregistrés et inclut deux boutons pour **sauvegarder les tweets dans une base de données** et **réentraîner le modèle** (implémentation à venir).

### Routes principales

- **`/predict-only`** : Route utilisée pour soumettre des tweets réels et obtenir un sentiment.
- **`/send-message`** : Route utilisée sur la page de test pour envoyer des tweets et confirmer les prédictions.
- **`/confirm-sentiment`** : Permet de valider ou corriger une prédiction envoyée.
- **`/adm`** : Affiche les tweets et donne accès à l'administration pour sauvegarder ou réentraîner le modèle (fonctionnalités à venir).

## Monitoring avec Azure Application Insights

Azure Insights est utilisé pour suivre les métriques suivantes :
- **Prédictions positives et négatives** sur la page de prédiction.
- **Prédictions incorrectes** sur la page de test, lorsque l'utilisateur corrige la prédiction du modèle.

## Instructions d'installation et d'exécution

1. **Installation des dépendances** :  
   Exécuter la commande suivante pour installer les packages nécessaires :
   ```bash
   pip install -r requirements.txt
   ```
## Instructions d'installation et d'exécution

1. **Lancer l'API en local** :
   ```bash
   python api/main.py
   ```
2. **Accéder aux interfaces :**

```plaintext
Page de prédiction : http://localhost:5000/predict
Page de test : http://localhost:5000/test
Page d'administration : http://localhost:5000/adm
```
## Modèle utilisé

Le projet utilise plusieurs modèles, y compris :

Régression logistique pour une approche de base.
CNN, LSTM et BERT pour des modèles avancés.
Suivi des performances avec MLflow et Azure Application Insights pour surveiller les métriques en production.
