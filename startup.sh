#!/bin/bash

# Installation des dépendances définies dans requirements.txt
echo "Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

# Lancer l'application Flask avec Gunicorn sur le port spécifié
if [ -z "$PORT" ]; then
  PORT=8000
fi

echo "Lancement de l'application avec Gunicorn sur le port $PORT..."
gunicorn --bind 0.0.0.0:$PORT main:app
