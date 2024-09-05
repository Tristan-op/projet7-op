import os
import zipfile
import nltk
nltk.download('wordnet')

import tensorflow as tf
import csv
from flask import Flask, render_template, request, jsonify
from datetime import datetime

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# Initialiser le lemmatizer et le modèle FastText
lemmatizer = WordNetLemmatizer()
print("Chargement du modèle FastText...")

# Chemin vers le modèle FastText
model_path = './models/cc.en.300.bin'

# Vérifier si le modèle existe, sinon le télécharger
if not os.path.exists(model_path):
    print("Téléchargement du modèle FastText...")
    os.system("wget -P ./models/ https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz")
    os.system("gzip -d ./models/cc.en.300.bin.gz")  # Décompresse le fichier
    print("Téléchargement terminé.")

# Charger le modèle FastText
model = fasttext.load_model(model_path)

# Chemin du fichier .zip du modèle LSTM
zip_file_path = './models/LSTM_plus_Lemmatization_plus_FastText_model.zip'
extract_dir = './models/temp_model'

# Décompression et chargement du modèle LSTM si nécessaire
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Charger le modèle LSTM
model_path = os.path.join(extract_dir, 'LSTM_plus_Lemmatization_plus_FastText_model.h5')
try:
    lstm_model = tf.keras.models.load_model(model_path)
    print("Modèle LSTM chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle LSTM : {e}")

# Gestion des erreurs de chargement du modèle
try:
    lstm_model = tf.keras.models.load_model(model_path)
    print("Modèle LSTM chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle LSTM : {e}")

# Initialiser l'application Flask
app = Flask(__name__)

# Fonction pour enregistrer les tweets dans un fichier CSV
def save_tweet_to_csv(tweet_data):
    file_exists = os.path.isfile('tweets.csv')
    
    with open('tweets.csv', mode='a', newline='', encoding='utf-8') as file:
        fieldnames = ['ID', 'Nom Utilisateur', 'Heure', 'Tweet', 'Sentiment', 'Prédiction']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # Écrire les en-têtes si le fichier est créé pour la première fois

        writer.writerow(tweet_data)

# Route d'accueil pour l'API
@app.route("/", methods=["GET"])
def home():
    return render_template("welcome.html")

# Route pour afficher le formulaire de tweet
@app.route("/continue", methods=["GET"])
def continue_to_tweet():
    return render_template("tweet.html")

# Route pour soumettre un tweet et comparer les résultats
@app.route("/submit_tweet", methods=["POST"])
def submit_tweet():
    name = request.form['name']
    tweet = request.form['tweet']
    sentiment_value = int(request.form['sentiment'])  # 0 pour positif, 1 pour négatif

    # Prétraiter le tweet (lemmatisation)
    processed_tweet = lemmatizer.lemmatize(tweet)

    # Utiliser le modèle FastText pour vectoriser le tweet
    tweet_vector = model.get_sentence_vector(processed_tweet)

    # Faire une prédiction avec le modèle LSTM
    prediction = lstm_model.predict([tweet_vector])

    # Si la prédiction est > 0.5, alors le modèle prédit "négatif" (1), sinon "positif" (0)
    predicted_sentiment = 1 if prediction[0] > 0.5 else 0

    # Comparer la prédiction du modèle avec le sentiment soumis par l'utilisateur
    is_correct = (predicted_sentiment == sentiment_value)

    # Préparer les données à sauvegarder dans le fichier CSV
    tweet_data = {
        'ID': sum(1 for _ in open('tweets.csv')) if os.path.exists('tweets.csv') else 1,
        'Nom Utilisateur': name,
        'Heure': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Tweet': tweet,
        'Sentiment': sentiment_value,
        'Prédiction': predicted_sentiment
    }

    # Sauvegarder le tweet dans le fichier CSV 
    save_tweet_to_csv(tweet_data)

    # Message à retourner à l'utilisateur
    result_message = f"Merci {name}, votre tweet a été soumis avec succès."
    if is_correct:
        result_message += " Le modèle a correctement prédit le sentiment."
    else:
        result_message += " Le modèle n'a pas correctement prédit le sentiment."

    return jsonify({
        'name': name,
        'tweet': tweet,
        'sentiment_utilisateur': sentiment_value,
        'sentiment_model': predicted_sentiment,
        'correct_prediction': is_correct,
        'message': result_message
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
