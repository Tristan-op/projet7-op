#!/bin/bash

echo "Importation des modules pour les tests..."
python - << END
import numpy as np
import tensorflow as tf
import spacy
import gensim.downloader as api

END

echo "Lancement de l'application avec Gunicorn..."

