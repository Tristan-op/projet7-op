#!/bin/bash
echo "Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Lancement de l'application avec Gunicorn..."
