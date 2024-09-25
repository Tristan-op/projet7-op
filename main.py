import os
import subprocess
import sys

# Fonction pour installer les modules si non installés
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Installer FastAPI, Uvicorn, et autres dépendances critiques
try:
    import fastapi
except ImportError:
    install_package('fastapi')


# Télécharger et installer FastText précompilé
def install_fasttext_from_binary():
    print("Téléchargement de FastText précompilé...")
    fasttext_bin_url = "https://github.com/facebookresearch/fastText/archive/v0.9.2.zip"
    subprocess.run(["curl", "-L", "-o", "fasttext.zip", fasttext_bin_url])

    # Extraire l'archive
    subprocess.run(["unzip", "fasttext.zip"])

    # Naviguer dans le répertoire extrait
    os.chdir("fastText-0.9.2")

    # Compiler et installer FastText manuellement
    print("Compilation et installation de FastText...")
    subprocess.run(["make"])

    print("FastText installé avec succès.")

# Chemin vers le modèle FastText préentraîné
fasttext_model_path = "./cc.fr.300.bin"

# Étape 1: Télécharger et installer FastText s'il n'est pas installé
try:
    import fasttext
except ImportError:
    install_fasttext_from_binary()

# Étape 2: Télécharger le modèle FastText s'il n'est pas présent
if not os.path.exists(fasttext_model_path):
    print(f"Téléchargement du modèle FastText vers {fasttext_model_path}...")
    # Télécharger le modèle compressé
    subprocess.run(
        ["curl", "-o", fasttext_model_path + ".gz", "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.bin.gz"]
    )
    # Décompresser le fichier téléchargé
    subprocess.run(["gunzip", fasttext_model_path + ".gz"])
    print(f"Modèle FastText téléchargé et décompressé à {fasttext_model_path}")

# Charger le modèle FastText
try:
    ft_model = fasttext.load_model(fasttext_model_path)
    print(f"Modèle FastText chargé depuis {fasttext_model_path}")
except Exception as e:
    print(f"Erreur lors du chargement du modèle FastText: {e}")

# Maintenant, les modules sont installés, on peut les importer
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import spacy
import fasttext
import tensorflow as tf
import numpy as np
from datetime import datetime
import re

# Initialisation de l'application FastAPI
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Charger le modèle LSTM (fichier .h5)
lstm_model = tf.keras.models.load_model("./models/LSTM_plus_Lemmatization_plus_FastText_model.h5")

# Page d'accueil
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("welcome.html", {"request": request})

# Page de chat
@app.get("/chat")
async def chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

# Modèle de données pour les tweets
class TweetData(BaseModel):
    username: str
    message: str
    sentiment: int

# Analyse du tweet
@app.post("/analyze")
async def analyze_tweet(tweet: TweetData):
    # Charger le modèle de langue spaCy pour l'anglais
    nlp = spacy.load('en_core_web_sm')

    # Prétraitement du texte avec suppression des caractères spéciaux et mise en minuscule
    preprocessed_text = re.sub(r'[^\w\s]', '', tweet.message.lower())

    # Lemmatisation du texte avec spaCy
    doc = nlp(preprocessed_text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])

    # Prédiction avec FastText
    prediction, _ = ft_model.predict(lemmatized_text)

    # Comparaison avec le sentiment attendu
    if prediction[0] == str(tweet.sentiment):
        response = "J'ai bien compris tes sentiments."
    else:
        response = "Désolé, j'apprends encore, je n'ai pas bien compris tes sentiments."

    return {"message": response}