from flask import Flask, jsonify, request, render_template, redirect, url_for
from datetime import datetime
from threading import Thread
import time
import gensim.downloader as api
import spacy

app = Flask(__name__, template_folder="templates")

# Simuler une base de données en mémoire pour stocker les messages
messages = []

# Variables globales pour le modèle et l'état de chargement
ft_model = None
nlp = None
models_loaded = False

# Fonction pour charger le modèle en arrière-plan
def load_models():
    global ft_model, nlp, models_loaded
    try:
        print("Chargement des modèles FastText et spaCy en cours...")
        ft_model = api.load('fasttext-wiki-news-subwords-300')
        nlp = spacy.load('en_core_web_sm')
        models_loaded = True
        print("Modèles FastText et spaCy chargés.")
    except Exception as e:
        print(f"Erreur lors du chargement des modèles : {e}")

# Page de préchargement des modèles
@app.route('/')
def preload():
    if models_loaded:
        return redirect(url_for('home'))  # Rediriger vers la page d'accueil une fois les modèles chargés
    return render_template("preload.html")  # Afficher la page de préchargement

# Page d'accueil
@app.route('/welcome', methods=['GET'])
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

        # Ici, vous pouvez ajouter le traitement du modèle LSTM si nécessaire

        # Stocker les messages dans la liste avec les informations de temps
        messages.append({
            'username': username,
            'message': message,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sentiment': sentiment
        })

        return jsonify({'result': "Message reçu."}), 200
    else:
        return jsonify({'result': 'Erreur lors de l\'envoi du message'}), 400

@app.route('/chat-history', methods=['GET'])
def chat_history():
    return jsonify({'messages': [{'username': msg['username'], 'message': msg['message'], 'time': msg['time']} for msg in messages]})

# Page pour l'administrateur afin de voir la liste des messages avec le sentiment et l'heure
@app.route('/adm', methods=['GET'])
def admin_view():
    return render_template("adm.html", messages=messages)

if __name__ == '__main__':
    # Créer un thread pour charger les modèles en arrière-plan
    model_thread = Thread(target=load_models)
    model_thread.start()

    # Démarrer l'application Flask
    app.run(debug=True)
