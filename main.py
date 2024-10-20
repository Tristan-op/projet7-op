from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import re
from datetime import datetime
from applicationinsights import TelemetryClient  # Import Azure Insights SDK

# Initialisation de l'application Flask
app = Flask(__name__)

# Charger le modèle et le CountVectorizer depuis le fichier pickle
model_path = './models/best_model_with_vectorizer.pkl'
with open(model_path, 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    count_vectorizer = data['vectorizer']

# Charger le modèle Spacy
def load_spacy_model():
    try:
        nlp = spacy.load('./models/spacy_model')
        print("Modèle Spacy chargé depuis ./models/spacy_model")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle Spacy : {e}")
        raise e
    return nlp

nlp = load_spacy_model()

# Fonction de prétraitement du texte
def preprocess_text(text):
    """ Prétraitement du texte : nettoyage, lemmatisation """
    text = re.sub(r'[^\w\s]', '', text.lower())  # Nettoyer les caractères spéciaux
    doc = nlp(text)
    lemmatized_words = [token.lemma_ for token in doc]
    return ' '.join(lemmatized_words)

# Initialisation de TelemetryClient pour Application Insights
tc = TelemetryClient('9ea1ad3d-2949-4e6f-a84b-4555cb14bd23')

# Simuler une base de données en mémoire pour stocker les messages
messages = []

@app.route('/')
def home():
    return render_template("welcome.html")

@app.route('/chat')
def chat():
    return render_template("chat.html", messages=messages)

@app.route('/chat-history', methods=['GET'])
def chat_history():
    return jsonify({'messages': messages})

@app.route('/send-message', methods=['POST'])
def send_message():
    try:
        data = request.get_json()

        # Vérifier que les champs nécessaires sont présents
        if 'username' not in data or 'message' not in data:
            return jsonify({'result': 'Données invalides, certains champs manquent'}), 400

        username = data['username']
        message = data['message']

        # Prétraitement du message avec la lemmatisation
        preprocessed_message = preprocess_text(message)

        # Vectorisation du message
        message_vect = count_vectorizer.transform([preprocessed_message])

        # Prédiction du modèle de régression
        prediction = model.predict(message_vect)
        predicted_sentiment = 1 if prediction[0] > 0.5 else 0

        # Demander à l'utilisateur de valider la prédiction
        response = f"Le modèle a prédit que votre message est {'positif' if predicted_sentiment == 1 else 'négatif'}. Êtes-vous d'accord ?"

        # Stocker le message de l'utilisateur dans une base de données simulée
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
        print(f"Erreur lors de l'envoi du message : {e}")
        return jsonify({'result': 'Erreur lors de l\'envoi du message'}), 500

@app.route('/confirm-sentiment', methods=['POST'])
def confirm_sentiment():
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
        if confirmation:
            tc.track_event('CorrectPrediction', {'username': username, 'message': message})
            tc.track_metric('CorrectPredictions', 1)
        else:
            tc.track_event('IncorrectPrediction', {'username': username, 'message': message})
            tc.track_metric('IncorrectPredictions', 1)

        tc.flush()
        return jsonify({'result': 'Merci pour la confirmation' if confirmation else 'Merci pour la correction'}), 200

    except Exception as e:
        print(f"Erreur lors de la confirmation : {e}")
        return jsonify({'result': 'Erreur lors de la confirmation'}), 500

if __name__ == '__main__':
    app.run(debug=True)
