from flask import Flask, render_template, request, jsonify, redirect
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

# Variables pour suivre les prédictions correctes
total_predictions = 0
correct_predictions = 0

# Simuler deux bases de données en mémoire avec des listes
tweets = []  # Pour la page predict.html (tweets réels)
tweets_test = []  # Pour les tests sur la page chat.html (tweets de test)

# Page d'accueil avec redirection vers predict.html
@app.route('/')
def home():
    return render_template("welcome.html")

@app.route('/redirect-predict', methods=['POST'])
def redirect_to_predict():
    return redirect('/predict')

# Page de prédiction (predict.html)
@app.route('/predict-only', methods=['POST'])
def predict_only():
    try:
        data = request.get_json()
        tweet = data['tweet']

        # Prétraitement du tweet
        preprocessed_tweet = preprocess_text(tweet)

        # Vectorisation du tweet
        tweet_vect = count_vectorizer.transform([preprocessed_tweet])

        # Prédiction du modèle
        prediction = model.predict(tweet_vect)
        predicted_sentiment = 'Positif' if prediction[0] > 0.5 else 'Négatif'

        # Sauvegarder dans la liste de tweets (sans confirmation)
        tweets.append({
            'username': data.get('username', 'Utilisateur'),  # Si un nom d'utilisateur est fourni
            'message': tweet,
            'sentiment': predicted_sentiment,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        # Envoi à Azure Insights pour les prédictions positives/négatives
        tc.track_event(f'Tweet_{predicted_sentiment}', {'tweet': tweet}, {'sentiment': 1 if predicted_sentiment == 'Positif' else 0})
        tc.flush()

        return jsonify({'sentiment': predicted_sentiment}), 200
    except Exception as e:
        return jsonify({'error': 'Erreur lors de la prédiction'}), 500

# Page des tests avec confirmation (chat.html)
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
        predicted_sentiment = 'Positif' if prediction[0] > 0.5 else 'Négatif'

        # Sauvegarder toutes les colonnes dans la liste de tests (tweet_test)
        tweets_test.append({
            'username': username,
            'message': message,
            'predicted_sentiment': predicted_sentiment,
            'sentiment': None,  # À confirmer par l'utilisateur
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        # Générer la réponse de validation
        response = f"S.A.R.A : Le modèle a prédit que votre message est {predicted_sentiment}. Êtes-vous d'accord ?"

        tc.flush()
        return jsonify({'result': response, 'prediction': predicted_sentiment}), 200

    except Exception as e:
        return jsonify({'result': 'Erreur lors de l\'envoi du message'}), 500

# Page Admin (adm.html) pour afficher les deux listes de tweets
@app.route('/adm')
def admin_view():
    return render_template("adm.html", tweets=tweets, tweets_test=tweets_test)

# Route pour afficher l'historique des messages de chat
@app.route('/chat-history', methods=['GET'])
def chat_history():
    return jsonify({'messages': tweets_test})

# Confirmation du sentiment
@app.route('/confirm-sentiment', methods=['POST'])
def confirm_sentiment():
    global total_predictions, correct_predictions
    try:
        data = request.get_json()
        username = data['username']
        message = data['message']
        confirmation = data['confirmation']

        # Ne pas modifier les anciens messages confirmés
        for msg in tweets_test:
            if msg['username'] == username and msg['message'] == message and msg['sentiment'] is None:
                if confirmation:
                    msg['sentiment'] = f"L'utilisateur a confirmé que le sentiment est {msg['predicted_sentiment'].lower()}"
                else:
                    msg['sentiment'] = f"L'utilisateur n'est pas d'accord avec la prédiction"
                    # Capture des erreurs de prédiction dans Azure Insights
                    tc.track_event('IncorrectPrediction', {'username': username, 'message': message, 'predicted_sentiment': msg['predicted_sentiment']})

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

        return jsonify({'message': 'Merci pour la confirmation' if confirmation else 'Merci pour la correction'}), 200

    except Exception as e:
        return jsonify({'message': 'Erreur lors de la confirmation'}), 500

if __name__ == '__main__':
    app.run(debug=True)
