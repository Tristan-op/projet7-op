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


END

echo "Lancement de l'application avec Gunicorn..."
gunicorn --bind 0.0.0.0:$PORT main:app
