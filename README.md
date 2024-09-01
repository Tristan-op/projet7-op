# Projet7-OP: Analyse de Sentiment pour Air Paradis

## Description du projet

Ce projet vise à développer un modèle d'intelligence artificielle capable de prédire le sentiment exprimé dans les tweets concernant la compagnie aérienne "Air Paradis". Le projet a pour objectif de créer un prototype fonctionnel qui peut être déployé sous forme d'API, permettant à "Air Paradis" d'anticiper les bad buzz sur les réseaux sociaux.

## Objectifs

- **Préparer et analyser les données** : Utiliser des données open source pour explorer et préparer les tweets pour la modélisation.
- **Développer plusieurs modèles** : Comparer une approche simple de régression logistique avec des modèles avancés, y compris un modèle basé sur BERT.
- **Déploiement d'une API** : Déployer l'API de prédiction sur une plateforme Cloud en utilisant les principes MLOps pour assurer un suivi et un déploiement continu.
- **Suivi de la performance** : Mettre en place un système de monitoring et d'alertes pour suivre la performance du modèle en production.

## Structure du projet

La structure du projet est organisée de manière à faciliter le développement, le déploiement, et le suivi du modèle d'analyse de sentiment.

```plaintext
projet7-op/
├── data/               # Dossier pour les jeux de données
├── notebooks/          # Dossier pour les notebooks Jupyter
├── scripts/            # Dossier pour les scripts Python
├── models/             # Dossier pour stocker les modèles entraînés
├── mlruns/             # Dossier pour les enregistrements MLFlow
├── tests/              # Dossier pour les tests unitaires
├── README.md           # Fichier de documentation du projet
├── requirements.txt    # Fichier pour lister les dépendances Python
├── .gitignore          # Fichier pour exclure certains fichiers du versioning
