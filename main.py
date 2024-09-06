import re
import spacy
import fasttext
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from datetime import datetime
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import os
import zipfile

# Charger le modèle spaCy pour la lemmatisation
nlp = spacy.load('fr_core_news_md')

# Charger le modèle FastText
fasttext_model = fasttext.load_model('./modèle/cc.fr.300.bin')

# Décompression du modèle de machine learning (LSTM) si non décompressé
if not os.path.exists('./modèle/sentiment_lstm_model.h5'):
    with zipfile.ZipFile('./modèle/sentiment_lstm_model.zip', 'r') as zip_ref:
        zip_ref.extractall('./modèle')

# Charger le modèle LSTM après décompression
lstm_model = load_model('./modèle/sentiment_lstm_model.h5')

# Initialiser l'application Flask
app = Flask(__name__)

# Fichier CSV pour stocker les tweets
CSV_FILE = 'tweets.csv'

# Initialiser le fichier CSV s'il n'existe pas encore
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=['username', 'message', 'sentiment', 'time'])
    df.to_csv(CSV_FILE, index=False)

# Fonction pour ajouter un message au fichier CSV
def add_message_to_csv(username, message, sentiment):
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = {'username': username, 'message': message, 'sentiment': sentiment, 'time': time}
    df = pd.read_csv(CSV_FILE)
    df = df.append(new_data, ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

# Prétraiter le texte
def preprocess_text(text):
    # Enlever les caractères spéciaux
    text = re.sub(r'\W', ' ', text)
    # Convertir en minuscules
    text = text.lower()
    # Lemmatisation avec spaCy
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

# Convertir le texte en vecteurs avec FastText
def text_to_fasttext_vector(text):
    words = text.split()
    word_vectors = [fasttext_model.get_word_vector(word) for word in words]
    return np.mean(word_vectors, axis=0)

# Fonction pour prédire le sentiment avec LSTM
def predict_sentiment(text):
    cleaned_text = preprocess_text(text)
    # Tokenisation et séquence si nécessaire (ajuster selon ton tokenizer)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded_seq = pad_sequences(seq, maxlen=100)
    prediction = lstm_model.predict(padded_seq)
    return np.argmax(prediction)  # 0 pour négatif, 1 pour positif

# Route pour la page Welcome
@app.route('/')
def welcome():
    return render_template('welcome.html')

# Route pour la page Chat
@app.route('/chat')
def chat():
    return render_template('chat.html')

# Endpoint pour envoyer un message
@app.route('/send-message', methods=['POST'])
def send_message():
    # Toujours "Bob" comme nom d'utilisateur
    username = "Bob"
    data = request.get_json()
    message = data['message']
    user_sentiment = int(data['sentiment'])  # 0 pour négatif, 1 pour positif

    # Prédire le sentiment avec le modèle
    predicted_sentiment = predict_sentiment(message)

    # Ajouter le message dans le fichier CSV
    add_message_to_csv(username, message, user_sentiment)

    # Comparer le sentiment prédit avec celui fourni par l'utilisateur
    if predicted_sentiment == user_sentiment:
        result_message = "Le sentiment a bien été prédit."
    else:
        result_message = "Le sentiment n'a pas été bien prédit."

    # Retourner la réponse JSON
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

if __name__ == '__main__':
    app.run(debug=True)
