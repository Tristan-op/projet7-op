from flask import Flask, jsonify, render_template
from threading import Thread
import time
import gensim.downloader as api
import spacy

app = Flask(__name__, template_folder="templates")

# Variables pour suivre l'état du chargement
loading_progress = {
    "fasttext_loaded": False,
    "spacy_loaded": False
}

# Simuler une base de données en mémoire pour stocker les messages
messages = []

# Fonction pour charger le modèle FastText
def load_fasttext_model():
    global loading_progress
    time.sleep(2)  # Simuler le temps de chargement
    ft_model = api.load('fasttext-wiki-news-subwords-300')
    loading_progress["fasttext_loaded"] = True
    print("Modèle FastText chargé.")

# Fonction pour charger le modèle spaCy
def load_spacy_model():
    global loading_progress
    time.sleep(2)  # Simuler le temps de chargement
    nlp = spacy.load('en_core_web_sm')
    loading_progress["spacy_loaded"] = True
    print("Modèle spaCy initialisé.")

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

# Page pour l'administrateur afin de voir la liste des messages
@app.route('/adm', methods=['GET'])
def admin_view():
    return render_template("adm.html", messages=messages)

# API pour vérifier l'état du chargement des modèles
@app.route('/check-progress', methods=['GET'])
def check_progress():
    total_tasks = 2
    completed_tasks = sum(loading_progress.values())
    progress_percentage = (completed_tasks / total_tasks) * 100
    return jsonify({
        "progress": progress_percentage,
        "completed": loading_progress
    })

# Démarrer le chargement des modèles FastText et spaCy en arrière-plan
if __name__ == '__main__':
    # Créer des threads pour charger les modèles en arrière-plan
    fasttext_thread = Thread(target=load_fasttext_model)
    spacy_thread = Thread(target=load_spacy_model)

    fasttext_thread.start()
    spacy_thread.start()

    app.run(debug=True)
