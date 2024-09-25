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

# Initialiser l'application FastAPI
app = FastAPI()

# Configuration des templates
templates = Jinja2Templates(directory="template")

# Charger le modèle TensorFlow
model = tf.keras.models.load_model("./models/LSTM_plus_Lemmatization_plus_FastText_model.h5")

# Charger le modèle FastText
ft_model = fasttext.load_model("cc.en.300.bin")

# Charger SpaCy pour la lemmatisation anglaise
nlp = spacy.load("en_core_web_sm")

# Simuler un historique de chat (pour test)
chat_history = []

class MessageRequest(BaseModel):
    message: str
    sentiment: int  # 0 = négatif, 1 = positif

def preprocess_text(text: str):
    text = text.lower()  # Convertir en minuscules
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Supprimer les caractères spéciaux
    doc = nlp(text)  # Lemmatisation
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    words = lemmatized_text.split()
    vectors = np.array([ft_model.get_word_vector(word) for word in words])
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)  # Moyenne des vecteurs FastText
    return np.zeros(300)

@app.get("/")
async def get_welcome_page(request: Request):
    return templates.TemplateResponse("welcome.html", {"request": request})

@app.get("/continue")
async def continue_to_chat():
    return RedirectResponse("/chat")

@app.get("/exit")
async def exit_app():
    return {"message": "Vous avez quitté l'application."}

@app.get("/chat")
async def get_chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/chat-history")
async def get_chat_history():
    return {"messages": chat_history}

@app.post("/send-message")
async def send_message(message: MessageRequest):
    # Prétraiter le texte du message
    processed_text = preprocess_text(message.message)
    processed_text = np.expand_dims(np.expand_dims(processed_text, axis=0), axis=0)

    # Prédire le sentiment avec le modèle
    predicted_sentiment = model.predict(processed_text)
    predicted_sentiment = int(predicted_sentiment > 0.5)

    # Vérifier si le sentiment est correct
    if predicted_sentiment == message.sentiment:
        result = "L'IA a bien compris tes sentiments."
    else:
        result = "Désolé, l'IA n'a pas bien compris tes sentiments."

    # Ajouter le message à l'historique des chats
    chat_history.append({
        "username": "User",
        "message": message.message,
        "sentiment": message.sentiment,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
