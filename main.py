from flask import Flask, jsonify, request, render_template
from datetime import datetime

app = Flask(__name__, template_folder="templates")

# Simuler une base de données en mémoire pour stocker les messages
messages = []

@app.route('/')
def home():
    return render_template("welcome.html")

@app.route('/chat', methods=['GET'])
def chat():
    return render_template("chat.html")

@app.route('/send-message', methods=['POST'])
def send_message():
    data = request.get_json()

    # Vérification que les champs sont remplis
    if not data or 'username' not in data or 'message' not in data or 'sentiment' not in data:
        return jsonify({"error": "Tous les champs doivent être remplis"}), 400

    # Enregistrer le message avec l'heure
    new_message = {
        "username": data['username'],
        "message": data['message'],
        "sentiment": int(data['sentiment']),  # Stockage du sentiment (1 = positif, 0 = négatif)
        "time": datetime.now().strftime("%H:%M:%S")
    }
    messages.append(new_message)

    return jsonify({"result": "Message envoyé avec succès"})

@app.route('/chat-history', methods=['GET'])
def chat_history():
    return jsonify({"messages": messages})


