from flask import Flask, jsonify, request, render_template
from datetime import datetime

app = Flask(__name__, template_folder="templates")

# Simuler une base de données en mémoire pour stocker les messages
messages = []

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
    # Renvoie l'historique des messages sans le sentiment (pour l'affichage dans le chat)
    return jsonify({'messages': [{'username': msg['username'], 'message': msg['message'], 'time': msg['time']} for msg in messages]})

# Page pour l'administrateur afin de voir la liste des messages avec le sentiment et la prédiction
@app.route('/adm', methods=['GET'])
def admin_view():
    # La page admin montre le message avec le sentiment et l'heure d'envoi
    return render_template("adm.html", messages=messages)