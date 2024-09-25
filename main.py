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

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Chargement du modèle FastText
ft_model = fasttext.load_model("./models/LSTM_plus_Lemmatization_plus_FastText_model.h5")

# Endpoint pour la page d'accueil
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("welcome.html", {"request": request})

# Endpoint pour la page de chat
@app.get("/chat")
async def chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

# Modèle de données pour les requêtes
class TweetData(BaseModel):
    username: str
    message: str
    sentiment: int

# Prétraitement du texte avec spaCy et FastText
@app.post("/analyze")
async def analyze_tweet(tweet: TweetData):
    # Traitement du texte ici avec spacy et fasttext
    preprocessed_text = re.sub(r'[^\w\s]', '', tweet.message.lower())
    lemmatized_text = spacy.load('en_core_web_sm')(preprocessed_text).lemma_
    
    # Prédiction avec FastText et TensorFlow
    prediction = ft_model.predict(lemmatized_text)

    # Comparaison avec le sentiment attendu
    if prediction == tweet.sentiment:
        response = "J'ai bien compris tes sentiments."
    else:
        response = "Désolé, j'apprends encore, je n'ai pas bien compris tes sentiments."
    
    return {"message": response}
