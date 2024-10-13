from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import spacy  # Bibliothèque pour la lemmatisation
import re
from datetime import datetime

app = Flask(__name__)



# Charger le modèle et le CountVectorizer depuis le fichier pickle
model_path = './models/best_model_with_vectorizer.pkl'
with open(model_path, 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    count_vectorizer = data['vectorizer']

def load_spacy_model():
    try:
        # Charger le modèle s'il est déjà installé
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        # Si le modèle n'est pas trouvé, le télécharger et le charger
        print("Le modèle 'en_core_web_sm' n'a pas été trouvé. Téléchargement en cours...")
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load('en_core_web_sm')
    return nlp



def preprocess_text(text):
    """ Prétraitement du texte : nettoyage, lemmatisation """
    text = re.sub(r'[^\w\s]', '', text.lower())  # Nettoyer les caractères spéciaux
    doc = nlp(text)
    lemmatized_words = [token.lemma_ for token in doc]
    return ' '.join(lemmatized_words)

@app.route('/')
def home():
    return render_template("welcome.html")

@app.route('/chat')
def chat():
    # Passer les messages à la page chat
    return render_template("chat.html", messages=messages)

@app.route('/chat-history', methods=['GET'])
def chat_history():
    return jsonify({'messages': messages})


@app.route('/send-message', methods=['POST'])
def send_message():
    try:
        data = request.get_json()

        # Vérifier que les champs nécessaires sont présents
        if 'username' not in data or 'message' not in data or 'sentiment' not in data:
            return jsonify({'result': 'Données invalides, certains champs manquent'}), 400

        username = data['username']
        message = data['message']
        sentiment = int(data['sentiment'])  # Le sentiment doit être un entier (0 ou 1)

        # Prétraitement du message avec la lemmatisation
        preprocessed_message = preprocess_text(message)

        # Vectorisation du message avec le CountVectorizer ajusté
        message_vect = count_vectorizer.transform([preprocessed_message])

        # Faire la prédiction avec le modèle de régression logistique
        prediction = model.predict(message_vect)
        predicted_sentiment = 1 if prediction[0] > 0.5 else 0  # Si la prédiction est supérieure à 0.5, c'est "Positif"

        # Comparer le sentiment prédit avec le sentiment fourni
        if predicted_sentiment == sentiment:
            response = "J'ai bien compris tes sentiments."
        else:
            response = "Désolé, je n'ai pas bien compris tes sentiments."

        # Stocker le message de l'utilisateur dans une base de données simulée
        messages.append({
            'username': username,
            'message': message,
            'sentiment': 'Positif' if sentiment == 1 else 'Négatif',  # Sentiment donné par l'utilisateur
            'predicted_sentiment': 'Positif' if predicted_sentiment == 1 else 'Négatif',  # Prédiction du modèle
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        # Stocker également la réponse du modèle avec le nom "S.A.R.A"
        messages.append({
            'username': 'S.A.R.A',
            'message': response,
            'sentiment': '',
            'predicted_sentiment': '',
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        return jsonify({'result': response}), 200

    except Exception as e:
        print(f"Erreur lors de l'envoi du message : {e}")
        return jsonify({'result': 'Erreur lors de l\'envoi du message'}), 500

# Simuler une base de données en mémoire pour stocker les messages
messages = []
@app.route('/adm')
def admin_view():
    # Passer les messages à la page d'administration
    return render_template("adm.html", messages=messages)

@app.route('/exit')
def exit_app():
    return "Merci d'avoir utilisé l'application. L'application est maintenant fermée."

if __name__ == '__main__':
    app.run(debug=True)
