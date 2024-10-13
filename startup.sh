#!/bin/bash

echo "Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Importation des modules pour les tests..."
python - << END
import numpy as np
import tensorflow as tf
import spacy
import gensim.downloader as api
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Activer l'environnement virtuel
source venv/bin/activate

# Télécharger le modèle SpaCy si non présent
python -m spacy download en_core_web_sm


END
source venv/bin/activate
echo "Lancement de l'application avec Gunicorn..."

gunicorn --bind 0.0.0.0:$PORT main:app
