#!/bin/bash
pip install spacy
python -m spacy download en_core_web_sm

# Utiliser la variable d'environnement PORT définie par Azure, sinon utiliser le port 80
PORT=${PORT:-80}

uvicorn main:app --host 0.0.0.0 --port $PORT &
