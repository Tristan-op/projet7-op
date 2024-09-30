#!/bin/bash

# Installer les dépendances définies dans requirements.txt
echo "Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

# Vérification et définition du port par défaut si $PORT n'est pas défini
if [ -z "$PORT" ]; then
  PORT=8000
  echo "Port non défini. Utilisation du port par défaut : $PORT"
fi

# Démarrer l'application Flask avec Gunicorn en arrière-plan
echo "Démarrage de l'application avec Gunicorn sur le port $PORT..."
gunicorn --bind 0.0.0.0:$PORT main:app &
