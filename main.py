import os
import zipfile
import re
import mlflow.keras
import numpy as np
import pandas as pd
import fasttext
from flask import Flask, request, jsonify, render_template
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.config import experimental  # Importer seulement les fonctions nécessaires pour la configuration

# Limiter l'utilisation des ressources
physical_devices = experimental.list_physical_devices('CPU')  # Utiliser uniquement les CPU
if physical_devices:
    try:
        for device in physical_devices:
            experimental.set_virtual_device_configuration(
                device,
                [experimental.VirtualDeviceConfiguration(memory_limit=int(0.8 * experimental.get_memory_info(device)["total_memory"]))]
            )
    except RuntimeError as e:
        print(e)

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
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

# Prétraiter le texte
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Convertir le texte en vecteurs avec FastText
def text_to_fasttext_vector(text):
    words = text.split()
    word_vectors = [fasttext_model.get_word_vector(word) for word in words]
    return np.mean(word_vectors, axis=0)

# Fonction pour prédire le sentiment avec LSTM
def predict_sentiment(text):
    cleaned_text = preprocess_text(text)
    vectorized_text = text_to_fasttext_vector(cleaned_text)
    vectorized_text = np.expand_dims(vectorized_text, axis=0)
    prediction = lstm_model.predict(vectorized_text)
    return np.argmax(prediction)

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
    username = "Bob"
    data = request.get_json()
    message = data['message']
    user_sentiment = int(data['sentiment'])

    predicted_sentiment = predict_sentiment(message)

    add_message_to_csv(username, message, user_sentiment)

    if predicted_sentiment == user_sentiment:
        result_message = "Le sentiment a bien été prédit."
    else:
        result_message = "Le sentiment n'a pas été bien prédit."

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
    app.run(debug=False)
