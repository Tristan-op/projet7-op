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

try:
    import uvicorn
except ImportError:
    install_package('uvicorn')

try:
    import spacy
except ImportError:
    install_package('spacy')

try:
    import tensorflow as tf
except ImportError:
    install_package('tensorflow')

# Chemin vers le modèle FastText préentraîné
fasttext_model_path = "./cc.fr.300.bin"

# Étape 1: Télécharger le modèle FastText s'il n'est pas présent
if not os.path.exists(fasttext_model_path):
    print(f"Téléchargement du modèle FastText vers {fasttext_model_path}...")
    subprocess.run(
        ["curl", "-o", fasttext_model_path + ".gz", "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.bin.gz"]
    )
    subprocess.run(["gunzip", fasttext_model_path + ".gz"])  # Décompresser le fichier

# Étape 2: Vérifier si FastText est installé, sinon l'installer
try:
    import fasttext
except ImportError:
    install_package('fasttext')

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

# Charger le modèle FastText
ft_model = fasttext.load_model(fasttext_model_path)

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
    # Traitement du texte avec spaCy
    preprocessed_text = re.sub(r'[^\w\s]', '', tweet.message.lower())
    lemmatized_text = spacy.load('en_core_web_sm')(preprocessed_text).lemma_

    # Prédiction avec FastText
    prediction = ft_model.predict(lemmatized_text)

    # Comparaison avec le sentiment attendu
    if prediction == tweet.sentiment:
        response = "J'ai bien compris tes sentiments."
    else:
        response = "Désolé, j'apprends encore, je n'ai pas bien compris tes sentiments."

    return {"message": response}