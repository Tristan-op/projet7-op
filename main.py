from flask import Flask, jsonify, render_template
from threading import Thread
from datetime import datetime
import time
import numpy as np
import tensorflow as tf
import re
import gensim.downloader as api
import spacy

app = Flask(__name__, template_folder="templates")

# Variables pour suivre l'état du chargement
loading_progress = {
    "tensorflow_loaded": False,
    "spacy_loaded": False,
    "fasttext_loaded": False
}

# Simuler une base de données en mémoire pour stocker les messages
messages = []

# Charger le modèle TensorFlow (plus léger)
def load_tensorflow_model():
    global lstm_model
    lstm_model = tf.keras.models.load_model('./models/LSTM_plus_Lemmatization_plus_FastText_model.h5')
    loading_progress["tensorflow_loaded"] = True
    print("Modèle TensorFlow chargé.")

# Charger le modèle spaCy (modérément lourd)
def load_spacy_model():
    global nlp
    nlp = spacy.load('en_core_web_sm')
    loading_progress["spacy_loaded"] = True
    print("Modèle spaCy initialisé.")

# Charger le modèle FastText (plus lourd)
def load_fasttext_model():
    global ft_model
    ft_model = api.load('fasttext-wiki-news-subwords-300')
    loading_progress["fasttext_loaded"] = True
    print("Modèle FastText chargé.")

# Prétraitement du texte avec spaCy et nettoyage des caractères spéciaux
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

@app.route('/')
def loading():
    return render_template("loading.html")

@app.route('/welcome')
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
            'sentiment': sentiment,
            'predicted_sentiment': predicted_sentiment
        })

        return jsonify({'result': response}), 200
    else:
        return jsonify({'result': 'Erreur lors de l\'envoi du message'}), 400

@app.route('/chat-history', methods=['GET'])
def chat_history():
    return jsonify({'messages': [{'username': msg['username'], 'message': msg['message'], 'time': msg['time']} for msg in messages]})

# Page pour l'administrateur afin de voir la liste des messages avec le sentiment et la prédiction
@app.route('/adm', methods=['GET'])
def admin_view():
    return render_template("adm.html", messages=messages)

# API pour vérifier l'état du chargement des modèles
@app.route('/check-progress', methods=['GET'])
def check_progress():
    total_tasks = 3
    completed_tasks = sum(loading_progress.values())
    progress_percentage = (completed_tasks / total_tasks) * 100
    return jsonify({
        "progress": progress_percentage,
        "completed": loading_progress
    })

# Démarrer le chargement des modèles TensorFlow, spaCy et FastText en arrière-plan
if __name__ == '__main__':
    # Créer des threads pour charger les modèles en arrière-plan dans l'ordre : TensorFlow -> spaCy -> FastText
    tensorflow_thread = Thread(target=load_tensorflow_model)
    spacy_thread = Thread(target=load_spacy_model)
    fasttext_thread = Thread(target=load_fasttext_model)

    tensorflow_thread.start()
    tensorflow_thread.join()  # Attendre que TensorFlow se charge avant de lancer spaCy
    spacy_thread.start()
    spacy_thread.join()  # Attendre que spaCy se charge avant de lancer FastText
    fasttext_thread.start()

    app.run(debug=True)
