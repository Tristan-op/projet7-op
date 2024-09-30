#!/bin/bash

# Mise à jour de pip
echo "Mise à jour de pip..."
pip install --upgrade pip

# Installation des dépendances à partir du fichier requirements.txt
echo "Installation des dépendances..."
pip install -r requirements.txt

# Lancer l'application avec Gunicorn
echo "Lancement de l'application avec Gunicorn..."
gunicorn --bind 0.0.0.0:$PORT main:app
