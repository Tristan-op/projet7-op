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

# Télécharger le modèle spaCy en_core_web_sm si nécessaire
try:
    spacy.load('en_core_web_sm')
except OSError:
    spacy.cli.download('en_core_web_sm')

# Charger le modèle FastText avec Gensim
api.load('fasttext-wiki-news-subwords-300')

END

echo "Lancement de l'application avec Gunicorn..."
gunicorn --bind 0.0.0.0:$PORT main:app
