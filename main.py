# Importation complète des modules
import fastapi
import pydantic
import spacy
import fasttext
import tensorflow as tf
import numpy as np
import re
from datetime import datetime
from fastapi import responses

# Créer une application FastAPI
app = fastapi.FastAPI()

# Configurer Jinja2 pour les templates
templates = fastapi.templating.Jinja2Templates(directory="templates")

# Modèle pour la gestion des données
class Message(pydantic.BaseModel):
    user: str
    text: str
    sentiment: str

# Charger les modèles de machine learning
nlp = spacy.load("en_core_web_sm")
ft_model = fasttext.load_model("cc.en.300.bin")
lstm_model = tf.keras.models.load_model('./models/LSTM_plus_Lemmatization_plus_FastText_model.h5')

# Fonction de prétraitement des textes
def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text.lower())  # Suppression des caractères spéciaux
    doc = nlp(text)  # Lemmatisation
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    return lemmatized_text

# Page d'accueil (GET request)
@app.get("/")
async def home(request: fastapi.Request):
    return templates.TemplateResponse("welcome.html", {"request": request})

# Page de chat (GET request)
@app.get("/chat")
async def chat(request: fastapi.Request):
    return templates.TemplateResponse("chat.html", {"request": request})

# API pour traiter un message et prédire le sentiment (POST request)
@app.post("/predict")
async def predict(request: fastapi.Request, user: str = fastapi.Form(...), text: str = fastapi.Form(...), sentiment: str = fastapi.Form(...)):
    # Prétraiter le texte
    cleaned_text = preprocess_text(text)
    
    # Conversion via FastText et prédiction via le modèle LSTM
    ft_vector = ft_model.get_sentence_vector(cleaned_text)
    prediction = lstm_model.predict(np.array([ft_vector]))
    
    # Comparaison du résultat avec le sentiment indiqué
    predicted_sentiment = "positif" if prediction[0][0] > 0.5 else "négatif"
    if predicted_sentiment == sentiment:
        return {"message": "J'ai bien compris tes sentiments."}
    else:
        return {"message": "Désolé, j'apprends encore, je n'ai pas bien compris tes sentiments."}

# Redirection vers la page de chat après la page d'accueil
@app.post("/redirect")
async def redirect_to_chat():
    return fastapi.responses.RedirectResponse(url="/chat")

