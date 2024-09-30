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

# Test d'importation réussi
print("Importation des modules réussie.")

# Initialiser spaCy
nlp = spacy.load('en_core_web_sm')
print("Modèle spaCy initialisé.")

# Charger un modèle TensorFlow (si applicable)
lstm_model = tf.keras.models.load_model('./models/LSTM_plus_Lemmatization_plus_FastText_model.h5')
print("Modèle TensorFlow chargé.")
END

echo "Lancement de l'application avec Gunicorn..."
gunicorn --bind 0.0.0.0:$PORT main:app
