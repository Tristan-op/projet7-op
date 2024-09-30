#!/bin/bash

# Installer les dépendances définies dans requirements.txt
echo "Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

# Lancer l'application Flask avec Gunicorn
echo "Lancement de l'application avec Gunicorn..."
gunicorn --bind 0.0.0.0:$PORT main:app
