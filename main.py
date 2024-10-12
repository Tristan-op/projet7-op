

from flask import Flask, jsonify, render_template, request
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
    "machine_learning_loaded": False,
    "spacy_loaded": False,
    "fasttext_loaded": False
}

# Simuler une base de données en mémoire pour stocker les messages
messages = []

# Variables globales pour les modèles
ft_model = None
nlp = None
tflite_interpreter = None

# Fonction pour charger tous les modèles en arrière-plan
def load_all_models():
    global tflite_interpreter
    try:
        # Charger le modèle TFLite (Machine Learning)
        print("Chargement du modèle TFLite...")
        tflite_interpreter = tf.lite.Interpreter(model_path='./notebooks/modèle_avancé/cnn_model_256_3_0.5.tflite')
        tflite_interpreter.allocate_tensors()
        loading_progress["machine_learning_loaded"] = True
        print("Modèle TFLite chargé.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle TFLite: {e}")

    try:
        # Charger le modèle spaCy
        print("Chargement du modèle spaCy...")
        global nlp
        nlp = spacy.load('en_core_web_sm')
        loading_progress["spacy_loaded"] = True
        print("Modèle spaCy initialisé.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle spaCy: {e}")

    try:
        # Charger le modèle FastText
        print("Chargement du modèle FastText...")
        global ft_model
        ft_model = api.load('fasttext-wiki-news-subwords-300')
        loading_progress["fasttext_loaded"] = True
        print("Modèle FastText chargé.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle FastText: {e}")

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

# Route pour la page de chargement
@app.route('/')
def loading():
    return render_template("loading.html")

# Route pour la page d'accueil après chargement
@app.route('/welcome')
def home():
    return render_template("welcome.html")

# Route pour continuer vers le chat
@app.route('/continue', methods=['GET'])
def continue_to_chat():
    return render_template("chat.html")

# Route pour quitter l'application
@app.route('/exit', methods=['GET'])
def exit_app():
    return "L'application est fermée."

# Route pour afficher le chat
@app.route('/chat', methods=['GET'])
def chat():
    return render_template("chat.html", messages=messages)

# Route pour envoyer un message
@app.route('/send-message', methods=['POST'])
def send_message():
    global tflite_interpreter, ft_model, nlp
    data = request.get_json()

    # Vérifier si tous les modèles sont chargés
    if not (loading_progress["machine_learning_loaded"] and loading_progress["spacy_loaded"] and loading_progress["fasttext_loaded"]):
        return jsonify({'result': 'Les modèles ne sont pas encore chargés. Veuillez patienter.'}), 503

    if 'username' in data and 'message' in data and 'sentiment' in data:
        username = data['username']
        message = data['message']
        sentiment = data['sentiment']

        # Prétraitement du message
        preprocessed_message = preprocess_text(message)

        # Créer l'embedding FastText pour le message
        words = preprocessed_message.split()
        embedding_matrix = create_embedding_matrix(words, ft_model)

        # Préparer les données pour le modèle TFLite
        max_length = 100  # La longueur maximale attendue par le modèle
        embedding_dim = 300  # Dimension des embeddings FastText

        if len(embedding_matrix) < max_length:
            # Si l'embedding est plus court que max_length, nous le complétons avec des zéros
            padding = np.zeros((max_length - len(embedding_matrix), embedding_dim))
            embedding_matrix = np.vstack([embedding_matrix, padding])
        else:
            # Si l'embedding est plus long, nous le tronquons à max_length
            embedding_matrix = embedding_matrix[:max_length]

        # Ajouter une dimension pour correspondre au batch size attendu par le modèle (1, max_length, embedding_dim)
        input_data = np.expand_dims(embedding_matrix, axis=0)

        # Préparer les entrées pour le modèle TFLite
        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()

        # Charger les données dans le modèle
        tflite_interpreter.set_tensor(input_details[0]['index'], input_data)

        # Exécuter la prédiction
        tflite_interpreter.invoke()

        # Récupérer la sortie du modèle
        prediction = tflite_interpreter.get_tensor(output_details[0]['index'])

        # Interpréter la sortie : 1 = positif, 0 = négatif
        predicted_sentiment = 1 if prediction[0][0] > 0.5 else 0

        # Comparer le sentiment prédit avec le sentiment attendu
        if predicted_sentiment == int(sentiment):
            response = "J'ai bien compris tes sentiments."
        else:
            response = "Désolé, j'apprends encore, je n'ai pas bien compris tes sentiments."

        # Stocker le message dans la liste des messages
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

# Route pour récupérer l'historique du chat
@app.route('/chat-history', methods=['GET'])
def chat_history():
    return jsonify({'messages': [{'username': msg['username'], 'message': msg['message'], 'time': msg['time']} for msg in messages]})

# Route pour la vue administrateur
@app.route('/adm', methods=['GET'])
def admin_view():
    return render_template("adm.html", messages=messages)

# API pour vérifier l'état du chargement des modèles
@app.route('/check-progress', methods=['GET'])
def check_progress():
    # Calculer le pourcentage basé sur le nombre de modèles chargés
    total_tasks = len(loading_progress)
    completed_tasks = sum(loading_progress.values())
    progress_percentage = int((completed_tasks / total_tasks) * 100)
    return jsonify({
        "progress": progress_percentage,
        "completed": loading_progress
    })

# Démarrer le chargement des modèles en arrière-plan
if __name__ == '__main__':
    # Créer un thread pour charger tous les modèles en arrière-plan
    model_thread = Thread(target=load_all_models)
    model_thread.start()

    # Démarrer l'application Flask
    app.run(debug=True)
