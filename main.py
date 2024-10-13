from flask import Flask, render_template, request, jsonify
import spacy
import re
import tensorflow as tf
import fasttext
import numpy as np
from datetime import datetime
from applicationinsights import TelemetryClient
import gensim.downloader as api

# Initialisation de l'application Flask
app = Flask(__name__)

# Charger le modèle CNN (TFLite) et FastText
def load_cnn_model():
    try:
        # Charger le modèle CNN TFLite
        cnn_model_path = './models/cnn_model_256_3_0.5.tflite'
        interpreter = tf.lite.Interpreter(model_path=cnn_model_path)
        interpreter.allocate_tensors()
        print("Modèle CNN TFLite chargé avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle CNN : {e}")
        raise e
    return interpreter

# Charger le modèle FastText depuis l'API Gensim
def load_fasttext_model():
    try:
        # Charger le modèle FastText depuis l'API Gensim
        ft_model = api.load('fasttext-wiki-news-subwords-300')
        print("Modèle FastText chargé depuis l'API Gensim 'fasttext-wiki-news-subwords-300'")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle FastText via l'API : {e}")
        raise e
    return ft_model

# Charger le modèle Spacy
def load_spacy_model():
    try:
        # Charger le modèle Spacy depuis le disque
        nlp = spacy.load('./models/spacy_model')
        print("Modèle Spacy chargé depuis ./models/spacy_model")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle Spacy : {e}")
        raise e
    return nlp

# Charger les modèles
cnn_model = load_cnn_model()
fasttext_model = load_fasttext_model()
nlp = load_spacy_model()

# Initialisation de TelemetryClient pour Application Insights
tc = TelemetryClient('9ea1ad3d-2949-4e6f-a84b-4555cb14bd23')

# Simuler une base de données en mémoire pour stocker les messages
messages = []

# Prétraitement du texte avec Spacy et FastText
def preprocess_and_vectorize(text):
    """ Prétraitement du texte : nettoyage, lemmatisation, vectorisation FastText """
    # Nettoyage du texte
    text = re.sub(r'[^\w\s]', '', text.lower())

    # Lemmatisation
    doc = nlp(text)
    lemmatized_words = [token.lemma_ for token in doc]

    # Vectorisation avec FastText
    vectors = []
    for word in lemmatized_words:
        if word in fasttext_model.key_to_index:  # Vérifie que le mot existe dans le modèle
            vectors.append(fasttext_model[word])  # Utilise le vecteur du mot
        else:
            vectors.append(np.zeros(300))  # Si le mot n'est pas trouvé, ajouter un vecteur nul

    # Si le texte est plus court que 100 mots, on le remplit avec des zéros
    max_len = 100
    if len(vectors) < max_len:
        vectors.extend([np.zeros(300)] * (max_len - len(vectors)))

    # Garder seulement les 100 premiers mots
    vectors = vectors[:max_len]

    return np.array(vectors, dtype=np.float32)

# Fonction pour faire une prédiction
def predict_sentiment(text):
    input_data = preprocess_and_vectorize(text).reshape(1, 100, 300)
    input_details = cnn_model.get_input_details()
    cnn_model.set_tensor(input_details[0]['index'], input_data)

    # Exécuter le modèle
    cnn_model.invoke()

    output_details = cnn_model.get_output_details()
    prediction = cnn_model.get_tensor(output_details[0]['index'])

    return prediction[0][0]

@app.route('/')
def home():
    return render_template("welcome.html")

@app.route('/chat')
def chat():
    return render_template("chat.html", messages=messages)

@app.route('/chat-history', methods=['GET'])
def chat_history():
    return jsonify({'messages': messages})

@app.route('/send-message', methods=['POST'])
def send_message():
    try:
        data = request.get_json()

        # Vérifier que les champs nécessaires sont présents
        if 'username' not in data or 'message' not in data or 'sentiment' not in data:
            return jsonify({'result': 'Données invalides, certains champs manquent'}), 400

        username = data['username']
        message = data['message']
        sentiment = int(data['sentiment'])  # Le sentiment doit être un entier (0 ou 1)

        # Prétraitement du message avec la lemmatisation et FastText
        preprocessed_message = preprocess_and_vectorize(message)

        # Faire la prédiction avec le modèle CNN
        prediction = predict_sentiment(message)
        predicted_sentiment = 1 if prediction > 0.5 else 0  # Si la prédiction est supérieure à 0.5, c'est "Positif"

        # Comparer le sentiment prédit avec le sentiment fourni
        if predicted_sentiment == sentiment:
            response = "J'ai bien compris tes sentiments."
            tc.track_event('CorrectPrediction', {'username': username, 'message': message}, {'sentiment': sentiment, 'predicted_sentiment': predicted_sentiment})
        else:
            response = "Désolé, je n'ai pas bien compris tes sentiments."
            tc.track_event('IncorrectPrediction', {'username': username, 'message': message}, {'sentiment': sentiment, 'predicted_sentiment': predicted_sentiment})

        # Incrémentation des prédictions correctes ou incorrectes
        if predicted_sentiment == sentiment:
            tc.track_metric('CorrectPredictions', 1)
        else:
            tc.track_metric('IncorrectPredictions', 1)

        # Stocker le message de l'utilisateur dans une base de données simulée
        messages.append({
            'username': username,
            'message': message,
            'sentiment': 'Positif' if sentiment == 1 else 'Négatif',  # Sentiment donné par l'utilisateur
            'predicted_sentiment': 'Positif' if predicted_sentiment == 1 else 'Négatif',  # Prédiction du modèle
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        # Stocker également la réponse du modèle avec le nom "S.A.R.A"
        messages.append({
            'username': 'S.A.R.A',
            'message': response,
            'sentiment': '',
            'predicted_sentiment': '',
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        # Envoyer les données à Azure Application Insights
        tc.flush()

        return jsonify({'result': response}), 200

    except Exception as e:
        print(f"Erreur lors de l'envoi du message : {e}")
        return jsonify({'result': 'Erreur lors de l\'envoi du message'}), 500

@app.route('/adm')
def admin_view():
    return render_template("adm.html", messages=messages)

@app.route('/exit')
def exit_app():
    return "Merci d'avoir utilisé l'application. L'application est maintenant fermée."

if __name__ == '__main__':
    app.run(debug=True)
