import os
import zipfile
import fasttext.util
import nltk
from flask import Flask, render_template, redirect, url_for
import tensorflow as tf

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# Initialiser le lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialiser le modèle FastText pré-entraîné
print("Chargement du modèle FastText...")
ft_model = fasttext.load_model('fasttext-wiki-news-subwords-300')

# Définir l'API Flask
app = Flask(__name__)

# Décompression du fichier .zip et chargement du modèle
zip_file_path = './models/LSTM_plus_Lemmatization_plus_FastText_model.zip'
extract_dir = './models/temp_model'

if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Charger le modèle .h5 décompressé
model_path = os.path.join(extract_dir, 'LSTM_plus_Lemmatization_plus_FastText_model.h5')

# Gestion des erreurs lors du chargement du modèle
try:
    lstm_model = tf.keras.models.load_model(model_path)
    print("Modèle LSTM chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle LSTM : {e}")

# Route d'accueil pour l'API avec message d'avertissement
@app.route("/", methods=["GET"])
def home():
    # Renvoyer la page HTML avec les boutons "OK" et "Exit"
    return render_template("welcome.html")

# Route pour continuer à l'étape de tweet
@app.route("/continue", methods=["GET"])
def continue_to_tweet():
    return "Vous pouvez maintenant tweeter via l'API (fonctionnalité à implémenter)."

# Route pour quitter l'application
@app.route("/exit", methods=["GET"])
def exit_application():
    return "Merci d'avoir utilisé notre service. L'application est fermée.", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))


