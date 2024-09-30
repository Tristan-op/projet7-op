from flask import Flask, render_template, redirect, url_for

# Créer l'application Flask
app = Flask(__name__, template_folder="templates")

# Définir la route pour la page d'accueil
@app.route('/')
def home():
    """
    Cette fonction répond à l'URL localhost:5000/
    Elle renvoie la page 'welcome.html'
    """
    return render_template('welcome.html')

# Route pour le bouton "OK" qui redirige vers la page chat
@app.route('/continue', methods=['GET'])
def go_to_chat():
    """
    Redirige l'utilisateur vers la page de chat lorsque le bouton 'OK' est cliqué.
    """
    return redirect(url_for('chat'))

# Route pour le bouton "Exit" qui ferme l'application
@app.route('/exit', methods=['GET'])
def exit_app():
    """
    Ferme l'application lorsque l'utilisateur clique sur le bouton 'Exit'.
    """
    return "Application terminée. Merci d'avoir utilisé notre service.", 200

# Définir la route pour la page "Chat"
@app.route('/chat')
def chat():
    """
    Cette fonction répond à l'URL localhost:5000/chat
    Elle renvoie la page 'chat.html'
    """
    return render_template('chat.html')

# Gérer l'envoi des messages
@app.route('/send-message', methods=['POST'])
def send_message():
    """
    Cette fonction gère l'envoi de message depuis le formulaire de la page chat.
    """
    data = request.get_json()
    
    # Vérifier si tous les champs sont remplis
    if 'username' not in data or 'message' not in data or 'sentiment' not in data:
        return jsonify({"result": "Tous les champs sont requis"}), 400

    # Simuler le traitement du message
    username = data['username']
    message = data['message']
    sentiment = data['sentiment']

    # Ajouter le message dans l'historique
    chat_history.append({
        "username": username,
        "message": message,
        "sentiment": sentiment,
        "time": "just now"
    })

    return jsonify({"result": "Message envoyé avec succès"}), 200

# Historique des messages
chat_history = []

@app.route('/chat-history', methods=['GET'])
def chat_history_view():
    """
    Cette fonction renvoie l'historique des messages en JSON.
    """
    return jsonify({"messages": chat_history})

