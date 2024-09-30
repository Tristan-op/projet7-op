import subprocess
import sys

# Installer manuellement la version compatible de NumPy si non installée
try:
    import numpy
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.23.5"])

# Installer manuellement la version compatible de SciPy si non installée
try:
    import scipy
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy==1.10.0"])



from flask import Flask, jsonify, request, render_template
from datetime import datetime
import numpy as np
import tensorflow as tf
import re
import spacy
import gensim.downloader as api

app = Flask(__name__, template_folder="templates")

# Simuler une base de données en mémoire pour stocker les messages
messages = []

# Charger le modèle FastText pré-entraîné via gensim
ft_model = api.load('fasttext-wiki-news-subwords-300')

# Charger le modèle LSTM (fichier .h5)
lstm_model = tf.keras.models.load_model("./models/LSTM_plus_Lemmatization_plus_FastText_model.h5")

# Initialiser spaCy pour la lemmatisation
nlp = spacy.load('en_core_web_sm')

@app.route('/')
def home():
    return render_template("welcome.html")

@app.route('/continue', methods=['GET'])
def continue_to_chat():
    return render_template("chat.html")

@app.route('/exit', methods=['GET'])
def exit_app():
    return "L'application est fermée."

@app.route('/chat', methods=['GET'])
def chat():
    return render_template("chat.html", messages=messages)

# Prétraitement du texte : lemmatisation, nettoyage des caractères spéciaux, etc.
def preprocess_text(text):
    # Convertir en minuscules et supprimer les caractères spéciaux
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Lemmatisation avec spaCy
    doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    
    return lemmatized_text

# Créer la matrice d'embedding avec FastText
def create_embedding_matrix(words, embedding_model, embedding_dim=300):
    embedding_matrix = np.zeros((len(words), embedding_dim))
    for idx, word in enumerate(words):
        if word in embedding_model:
            embedding_matrix[idx] = embedding_model[word]
        else:
            embedding_matrix[idx] = np.zeros(embedding_dim)
    return embedding_matrix

@app.route('/send-message', methods=['POST'])
def send_message():
    data = request.get_json()

    if 'username' in data and 'message' in data and 'sentiment' in data:
        username = data['username']
        message = data['message']
        sentiment = data['sentiment']

        # Prétraitement du message
        preprocessed_message = preprocess_text(message)

        # Embedding FastText
        words = preprocessed_message.split()
        embedding_matrix = create_embedding_matrix(words, ft_model)

        # Utiliser le modèle LSTM pour la prédiction
        prediction = lstm_model.predict(np.array([embedding_matrix]))  # Adapter la forme si nécessaire

        # Comparaison avec le sentiment attendu
        predicted_sentiment = 1 if prediction[0][0] > 0.5 else 0  # Seuil pour définir positif/négatif

        if predicted_sentiment == sentiment:
            response = "J'ai bien compris tes sentiments."
        else:
            response = "Désolé, j'apprends encore, je n'ai pas bien compris tes sentiments."

        # Stocker les messages dans la liste avec les informations de temps
        messages.append({
            'username': username,
            'message': message,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sentiment': sentiment
        })

        return jsonify({'result': response}), 200
    else:
        return jsonify({'result': 'Erreur lors de l\'envoi du message'}), 400

@app.route('/chat-history', methods=['GET'])
def chat_history():
    return jsonify({'messages': [{'username': msg['username'], 'message': msg['message'], 'time': msg['time']} for msg in messages]})

