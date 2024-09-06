import os
import zipfile
import re
import numpy as np
import pandas as pd
import fasttext
from flask import Flask, request, jsonify, render_template
from datetime import datetime
from tensorflow.keras.models import load_model

# --- Charger les modèles et initialiser le fichier CSV avant l'initialisation de Flask ---

# Charger le modèle FastText
fasttext_model = fasttext.load_model('./modèle/cc.fr.300.bin')

# Chemin du fichier zip et du modèle décompressé
zip_path = './models/LSTM_plus_Lemmatization_plus_FastText_model.zip'
extract_path = './models/unzipped_model/'
model_path = './models/unzipped_model/LSTM_plus_Lemmatization_plus_FastText_model.h5'

# Décompression du modèle de machine learning (LSTM)
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# Charger le modèle LSTM après décompression
if os.path.exists(model_path):
    lstm_model = load_model(model_path, compile=False)
else:
    raise FileNotFoundError(f"Le modèle n'a pas été trouvé à l'emplacement : {model_path}")

# Fichier CSV pour stocker les tweets
CSV_FILE = 'tweets.csv'

# Initialiser le fichier CSV s'il n'existe pas encore
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=['username', 'message', 'sentiment', 'time'])
    df.to_csv(CSV_FILE, index=False)

# --- Initialiser l'application Flask ---

app = Flask(__name__)

# --- Routes Flask ---

# Route pour la page Welcome
@app.route('/')
def welcome():
    return render_template('welcome.html')

# Route pour la page Chat
@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/send-message', methods=['POST'])
def send_message():
    data = request.get_json()  # Récupérer les données envoyées par l'utilisateur
    username = data.get('username', 'Utilisateur')  # Obtenir le nom d'utilisateur ou mettre un nom par défaut
    message = data['message']
    user_sentiment = int(data['sentiment'])

    # Prédire le sentiment avec le modèle
    predicted_sentiment = predict_sentiment(message)

    # Ajouter le vrai message et utilisateur dans le CSV
    add_message_to_csv(username, message, user_sentiment)

    # Toujours répondre avec le nom "Bob" et un message selon la prédiction du sentiment
    if predicted_sentiment == user_sentiment:
        result_message = "Bob a bien compris tes sentiments."
    else:
        result_message = "Désolé, Bob apprend encore, il n'a pas bien compris tes sentiments."

    # Réponse de l'API
    return jsonify({
        'message': 'Message enregistré avec succès',
        'result': result_message,
        'predicted_sentiment': predicted_sentiment,
        'user_sentiment': user_sentiment
    })

# Endpoint pour récupérer l'historique des tweets
@app.route('/chat-history', methods=['GET'])
def chat_history():
    df = pd.read_csv(CSV_FILE)
    messages = df.to_dict('records')
    return jsonify({'messages': messages})

# --- Démarrage de l'application ---
if __name__ == '__main__':
    app.run(debug=False)
