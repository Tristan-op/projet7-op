from flask import Flask, jsonify, request, render_template, redirect, url_for
from datetime import datetime

app = Flask(__name__, template_folder="templates")

# Simuler une base de données en mémoire pour stocker les messages
messages = []

@app.route('/')
def home():
    return render_template("welcome.html")

# Route pour rediriger vers la page chat lorsque l'utilisateur clique sur "OK"
@app.route('/continue', methods=['GET'])
def continue_to_chat():
    return redirect(url_for('chat'))

# Route pour quitter l'application (ferme simplement la fenêtre ou redirige)
@app.route('/exit', methods=['GET'])
def exit_app():
    
return "Application terminée. Merci d'avoir utilisé notre service."

@app.route('/chat', methods=['GET'])
def chat():
    return render_template("chat.html")

# Route pour gérer l'envoi de message
@app.route('/send-message', methods=['POST'])
def send_message():
    data = request.json
    username = data.get('username')
    message = data.get('message')
    sentiment = data.get('sentiment')

    if not username or not message or sentiment is None:
        return jsonify({"result": "Tous les champs doivent être remplis"}), 400

    # Stocker le message dans la "base de données" (liste Python pour l'instant)
    messages.append({
        'username': username,
        'message': message,
        'sentiment': sentiment,
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    return jsonify({"result": "Message envoyé avec succès"})

# Route pour récupérer l'historique du chat
@app.route('/chat-history', methods=['GET'])
def chat_history():
    return jsonify({"messages": messages})



