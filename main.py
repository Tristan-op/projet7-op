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
    import gensim
except ImportError:
    install_package('gensim')
try:
    import tensorflow as tf
except ImportError:
    install_package('tensorflow')

try:
    import spacy
except ImportError:
    install_package('spacy')
import os
import subprocess
import sys
import gensim.downloader as api
import numpy as np
import spacy
import tensorflow as tf
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import re

# Initialisation de l'application FastAPI
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Charger le modèle FastText pré-entraîné via gensim (aucune installation de fasttext nécessaire)
ft_model = api.load('fasttext-wiki-news-subwords-300')

# Fonction pour créer la matrice d'embeddings à partir de FastText
def create_embedding_matrix(tokenizer, embedding_model, embedding_dim=300):
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if word in embedding_model:
            embedding_matrix[i] = embedding_model[word]
    return embedding_matrix

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
    # Étape 1: Mettre en minuscules et supprimer les caractères spéciaux
    preprocessed_text = re.sub(r'[^\w\s]', '', tweet.message.lower())

    # Étape 2: Lemmatisation avec spaCy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(preprocessed_text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])

    # Étape 3: Création de la matrice d'embeddings FastText
    words = lemmatized_text.split()  # Séparer en tokens
    embedding_matrix = np.zeros((len(words), 300))  # 300 dimensions pour FastText
    for idx, word in enumerate(words):
        if word in ft_model:
            embedding_matrix[idx] = ft_model[word]
        else:
            embedding_matrix[idx] = np.zeros(300)  # Mot non trouvé dans FastText

    # Étape 4: Utiliser le modèle LSTM pour la prédiction
    prediction = lstm_model.predict(np.array([embedding_matrix]))  # Ajuster selon la forme du modèle LSTM

    # Comparaison avec le sentiment attendu
    if prediction == tweet.sentiment:
        response = "J'ai bien compris tes sentiments."
    else:
        response = "Désolé, j'apprends encore, je n'ai pas bien compris tes sentiments."

    return {"message": response}

