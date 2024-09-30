from flask import Flask, request, jsonify, render_template
import datetime

# Initialisation de l'application Flask
app = Flask(__name__, template_folder="templates")

# Variable pour stocker l'historique des messages
chat_history = []

# Page d'accueil (welcome.html doit être dans le dossier 'templates')
@app.route("/", methods=['GET'])
def home():
    return render_template("welcome.html")

# Page de chat (chat.html doit être dans le dossier 'templates')
@app.route("/chat", methods=['GET'])
def chat():
    return render_template("chat.html")

# Route pour gérer l'envoi des messages
@app.route("/send-message", methods=['POST'])
def send_message():
    data = request.get_json()

    # Vérification que tous les champs sont présents
    if not data or 'username' not in data or 'message' not in data or 'sentiment' not in data:
        return jsonify({"error": "Tous les champs doivent être remplis"}), 400

    # Récupérer les données envoyées
    username = data['username']
    message = data['message']
    sentiment = data['sentiment']

    # Ajouter le message à l'historique
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chat_history.append({
        "username": username,
        "message": message,
        "sentiment": sentiment,
        "time": timestamp
    })

    return jsonify({"result": "Message envoyé avec succès"})

# Route pour récupérer l'historique des messages
@app.route("/chat-history", methods=['GET'])
def chat_history_view():
    return jsonify({"messages": chat_history})

# Si on exécute ce fichier directement, lancer le serveur
if __name__ == '__main__':
    app.run(debug=True)
