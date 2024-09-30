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

        # Pour l'instant, on ne fait que stocker le message
        response = "Message reçu."

        # Stocker les messages dans la liste avec les informations de temps
        messages.append({
            'username': username,
            'message': message,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sentiment': sentiment
        })

        return jsonify({'result': response}), 200
    else:
        return jsonify({'result': 'Erreur lors de l\'envoi du message'}), 400

@app.route('/chat-history', methods=['GET'])
def chat_history():
    return jsonify({'messages': [{'username': msg['username'], 'message': msg['message'], 'time': msg['time']} for msg in messages]})

# Page pour l'administrateur afin de voir la liste des messages
@app.route('/adm', methods=['GET'])
def admin_view():
    return render_template("adm.html", messages=messages)