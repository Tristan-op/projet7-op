from flask import Flask, render_template, request, jsonify

# Créer l'application Flask
app = Flask(__name__, template_folder="templates")

# Définir la route pour la page d'accueil
@app.route('/')
def home():
    return render_template('welcome.html')

# Rediriger vers la page de chat via /continue
@app.route('/continue', methods=['GET'])
def go_to_chat():
    return render_template('chat.html')

# Route pour le bouton "Exit" qui ferme l'application
@app.route('/exit', methods=['GET'])
def exit_app():
    return "Application terminée. Merci d'avoir utilisé notre service.", 200

# Historique des messages
chat_history = []

# Route pour gérer l'envoi des messages
@app.route('/send-message', methods=['POST'])
def send_message():
    data = request.get_json()

    # Vérification des champs nécessaires
    if not data or 'username' not in data or 'message' not in data or 'sentiment' not in data:
        return jsonify({"result": "Erreur: Tous les champs sont requis"}), 400

    # Récupérer les données
    username = data['username']
    message = data['message']
    sentiment = data['sentiment']

    # Simuler l'ajout du message à l'historique
    chat_history.append({
        "username": username,
        "message": message,
        "sentiment": sentiment,
        "time": "just now"
    })

    return jsonify({"result": "Message envoyé avec succès"}), 200

# Route pour afficher l'historique des messages
@app.route('/chat-history', methods=['GET'])
def chat_history_view():
    return jsonify({"messages": chat_history})

# Si on exécute ce fichier directement, démarrer le serveur
if __name__ == '__main__':
    app.run(debug=True)
