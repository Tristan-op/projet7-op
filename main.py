import os
import subprocess
import sys

# Vérifier si pip est installé et essayer de l'installer si nécessaire
try:
    import pip
except ImportError:
    print("pip non installé, installation de pip...")
    subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])

# Installer les packages du fichier requirements.txt
def install_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
        except subprocess.CalledProcessError as e:
            print(f"Erreur lors de l'installation des dépendances: {e}")
    else:
        print("Le fichier requirements.txt est introuvable.")

# Appeler la fonction pour installer les dépendances
install_requirements()

# Ensuite, importer les bibliothèques après l'installation
import numpy as np
import pandas as pd
import os
import zipfile
import re
from flask import Flask, request, jsonify, render_template, redirect, url_for
from datetime import datetime
import fasttext
from tensorflow.keras.models import load_model
import spacy
from nltk.stem import WordNetLemmatizer
import csv


app = Flask(__name__)



# --- Charger les modèles FastText et le modèle de machine learning ---
fasttext_model = fasttext.load_model('./modèle/cc.fr.300.bin')

# Charger le modèle de machine learning (LSTM)
zip_path = './models/LSTM_plus_Lemmatization_plus_FastText_model.zip'
extract_path = './models/unzipped_model/'
model_path = './models/unzipped_model/LSTM_plus_Lemmatization_plus_FastText_model.h5'

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

lstm_model = load_model(model_path, compile=False)

# --- Initialiser l'application Flask ---
app = Flask(__name__)

# Fichier CSV pour stocker les tweets
CSV_FILE = 'tweets.csv'
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=['username', 'message', 'sentiment', 'time'])
    df.to_csv(CSV_FILE, index=False)

# --- Fonction de prétraitement du texte ---
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()  # Convertir en minuscules
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Supprimer les caractères spéciaux
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])  # Lemmatisation
    return text

# --- Prédire le sentiment ---
def predict_sentiment(message):
    processed_message = preprocess_text(message)
    vectorized_message = fasttext_model.get_sentence_vector(processed_message)
    vectorized_message = np.array([vectorized_message])  # Adapter à l'entrée du modèle LSTM
    prediction = lstm_model.predict(vectorized_message)
    predicted_sentiment = int(prediction > 0.5)  # Retourne 1 pour positif et 0 pour négatif
    return predicted_sentiment

# --- Sauvegarder le message dans le fichier CSV ---
def add_message_to_csv(username, message, sentiment):
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([username, message, sentiment, datetime.now()])

# --- Routes Flask ---

# Page d'accueil
@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/exit')
def exit_app():
    return "Merci d'avoir visité notre application !"

# Page de chat
@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/send-message', methods=['POST'])
def send_message():
    data = request.get_json()
    username = data.get('username', 'Utilisateur')
    message = data['message']
    user_sentiment = int(data['sentiment'])

    predicted_sentiment = predict_sentiment(message)

    # Sauvegarder le message et le sentiment
    add_message_to_csv(username, message, user_sentiment)

    # Comparaison du sentiment prédit et du sentiment utilisateur
    if predicted_sentiment == user_sentiment:
        result_message = "Bob a bien compris tes sentiments."
    else:
        result_message = "Désolé, Bob apprend encore, il n'a pas bien interprété tes sentiments."

    return jsonify({
        'message': 'Message enregistré avec succès',
        'result': result_message,
        'predicted_sentiment': predicted_sentiment,
        'user_sentiment': user_sentiment
    })

# Récupérer l'historique des chats
@app.route('/chat-history', methods=['GET'])
def chat_history():
    df = pd.read_csv(CSV_FILE)
    messages = df.to_dict('records')
    return jsonify({'messages': messages})

if __name__ == "__main__":
    app.run()

