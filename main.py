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

        # Prétraitement du message (désactivé pour le moment)
        # preprocessed_message = preprocess_text(message)

        # Embedding FastText (désactivé pour le moment)
        # words = preprocessed_message.split()
        # embedding_matrix = create_embedding_matrix(words, ft_model)

        # Utiliser le modèle LSTM pour la prédiction (désactivé pour le moment)
        # prediction = lstm_model.predict(np.array([embedding_matrix]))  # Adapter la forme si nécessaire

        # Simulation de la prédiction pour l'exemple (0 pour négatif, 1 pour positif)
        predicted_sentiment = sentiment  # Placeholder pour le vrai résultat de la prédiction

        # Stocker les messages dans la liste avec les informations de temps, sentiment et prédiction
        messages.append({
            'username': username,
            'message': message,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # Enregistrement de l'heure
            'sentiment': sentiment,  # Sentiment fourni par l'utilisateur
            'prediction': predicted_sentiment  # Prédiction simulée pour le moment
        })

        return jsonify({'result': "Message reçu."}), 200
    else:
        return jsonify({'result': 'Erreur lors de l\'envoi du message'}), 400

        return jsonify({'result': 'Erreur lors de l\'envoi du message'}), 400@app.route('/chat-history', methods=['GET'])
def chat_history():
    return jsonify({'messages': [{'username': msg['username'], 'message': msg['message'], 'time': msg['time']} for msg in messages]})

# Page pour l'administrateur afin de voir la liste des messages
@app.route('/adm', methods=['GET'])
def admin_view():
    return render_template("adm.html", messages=messages)