#!/bin/bash
echo "Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Vérification de l'installation de numpy..."
pip show numpy

echo "Lancement de l'application avec Gunicorn..."
