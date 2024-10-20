from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import re
from datetime import datetime
from applicationinsights import TelemetryClient

app = Flask(__name__)

# Charger le modèle et le CountVectorizer depuis le fichier pickle
model_path = './models/best_model_with_vectorizer.pkl'
with open(model_path, 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    count_vectorizer = data['vectorizer']

# Charger le modèle Spacy
nlp = spacy.load('./models/spacy_model')

# Fonction de prétraitement du texte
def preprocess_text(text):
    """Prétraitement du texte : nettoyage et lemmatisation"""
    text = re.sub(r'[^\w\s]', '', text.lower())
    doc = nlp(text)
    lemmatized_words = [token.lemma_ for token in doc]
    return ' '.join(lemmatized_words)

# Initialisation de TelemetryClient pour Application Insights
tc = TelemetryClient('9ea1ad3d-2949-4e6f-a84b-4555cb14bd23')

# Variables pour suivre le nombre total de prédictions et celles qui sont correctes
total_predictions = 0
correct_predictions = 0

# Simuler une base de données en mémoire pour stocker les messages
messages = []

@app.route('/')
def home():
    return render_template("welcome.html")

@app.route('/chat')
def chat():
    return render_template("chat.html", messages=messages)

@app.route('/adm')
def admin_view():
    return render_template("adm.html", messages=messages)

@app.route('/chat-history', methods=['GET'])
def chat_history():
    return jsonify({'messages': messages})

@app.route('/send-message', methods=['POST'])
def send_message():
    try:
        data = request.get_json()
        username = data['username']
        message = data['message']

        # Prétraitement du message avec la lemmatisation
        preprocessed_message = preprocess_text(message)

        # Vectorisation du message
        message_vect = count_vectorizer.transform([preprocessed_message])

        # Prédiction du modèle
        prediction = model.predict(message_vect)
        predicted_sentiment = 1 if prediction[0] > 0.5 else 0

        # Générer la réponse de validation
        response = f"S.A.R.A : Le modèle a prédit que votre message est {'positif' if predicted_sentiment == 1 else 'négatif'}. Êtes-vous d'accord ?"

        # Stocker le message
        messages.append({
            'username': username,
            'message': message,
            'predicted_sentiment': 'Positif' if predicted_sentiment == 1 else 'Négatif',
            'sentiment': None,  # À confirmer par l'utilisateur
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        tc.flush()
        return jsonify({'result': response, 'prediction': predicted_sentiment}), 200

    except Exception as e:
        return jsonify({'result': 'Erreur lors de l\'envoi du message'}), 500

@app.route('/confirm-sentiment', methods=['POST'])
def confirm_sentiment():
    global total_predictions, correct_predictions
    try:
        data = request.get_json()
        username = data['username']
        message = data['message']
        confirmation = data['confirmation']

        # Mettre à jour le sentiment avec la confirmation utilisateur
        for msg in messages:
            if msg['username'] == username and msg['message'] == message:
                msg['sentiment'] = 'Confirmé' if confirmation else 'Corrigé'

        # Traquer les événements corrects ou incorrects via Azure Insights
        total_predictions += 1
        if confirmation:
            correct_predictions += 1
            tc.track_event('CorrectPrediction', {'username': username, 'message': message})
        else:
            tc.track_event('IncorrectPrediction', {'username': username, 'message': message})

        # Calculer le pourcentage de prédictions correctes
        correct_percentage = (correct_predictions / total_predictions) * 100
        tc.track_metric('CorrectPredictionPercentage', correct_percentage)

        tc.flush()

        # Assurer que la réponse renvoie toujours un message
        return jsonify({'message': 'Merci pour la confirmation' if confirmation else 'Merci pour la correction'}), 200

    except Exception as e:
        return jsonify({'message': 'Erreur lors de la confirmation'}), 500

if __name__ == '__main__':
    app.run(debug=True)
