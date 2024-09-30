import os
import subprocess
import sys
import gensim.downloader as api
import numpy as np
import spacy
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import re

# Initialisation de l'application Flask
app = Flask(__name__)

# Charger le modèle FastText pré-entraîné via gensim (aucune installation de fasttext nécessaire)
ft_model = api.load('fasttext-wiki-news-subwords-300')

# Fonction pour créer la matrice d'embeddings à partir de FastText
def create_embedding_matrix(words, embedding_model, embedding_dim=300):
    embedding_matrix = np.zeros((len(words), embedding_dim))
    for idx, word in enumerate(words):
        if word in embedding_model:
            embedding_matrix[idx] = embedding_model[word]
        else:
            embedding_matrix[idx] = np.zeros(embedding_dim)  # Mot non trouvé dans FastText
    return embedding_matrix

# Charger le modèle LSTM (fichier .h5)
lstm_model = tf.keras.models.load_model("./models/LSTM_plus_Lemmatization_plus_FastText_model.h5")

# Page d'accueil (welcome.html doit être dans le dossier 'templates')
@app.route("/", methods=['GET'])
def home():
    return render_template("welcome.html")

# Page de chat (chat.html doit être dans le dossier 'templates')
@app.route("/chat", methods=['GET'])
def chat():
    return render_template("chat.html")

# Analyse du tweet (en POST)
@app.route("/analyze", methods=['POST'])
def analyze_tweet():
    data = request.get_json()

    # Vérification des champs nécessaires
    if not data or 'username' not in data or 'message' not in data or 'sentiment' not in data:
        return jsonify({"error": "Tous les champs doivent être remplis"}), 400

    # Récupérer les données
    username = data['username']
    message = data['message']
    sentiment = data['sentiment']

    # Étape 1: Mettre en minuscules et supprimer les caractères spéciaux
    preprocessed_text = re.sub(r'[^\w\s]', '', message.lower())

    # Étape 2: Lemmatisation avec spaCy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(preprocessed_text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])

    # Étape 3: Création de la matrice d'embeddings FastText
    words = lemmatized_text.split()  # Séparer en tokens
    embedding_matrix = create_embedding_matrix(words, ft_model)

    # Étape 4: Utiliser le modèle LSTM pour la prédiction
    prediction = lstm_model.predict(np.array([embedding_matrix]))  # Ajuster selon la forme du modèle LSTM

    # Comparaison avec le sentiment attendu
    if prediction == sentiment:
        response = "J'ai bien compris tes sentiments."
    else:
        response = "Désolé, j'apprends encore, je n'ai pas bien compris tes sentiments."

    return jsonify({"message": response})

