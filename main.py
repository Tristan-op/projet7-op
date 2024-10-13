from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem import PorterStemmer
import re

app = Flask(__name__)

# Charger le modèle et les composants de transformation
model_path = './models/Stemming_plus_CountVectorizer_plus_TF-IDF_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Initialiser les objets nécessaires (vectorizer, TF-IDF, etc.)
count_vectorizer = model['vectorizer']
tfidf_transformer = model['tfidf']
linear_model = model['model']

# Stemming
stemmer = PorterStemmer()

def preprocess_text(text):
    """ Prétraitement du texte : nettoyage, stemming """
    text = re.sub(r'[^\w\s]', '', text.lower())  # Nettoyer les caractères spéciaux
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

@app.route('/')
def home():
    return render_template("welcome.html")

@app.route('/chat')
def chat():
    return render_template("chat.html")

@app.route('/send-message', methods=['POST'])
def send_message():
    data = request.get_json()

    if 'username' in data and 'message' in data and 'sentiment' in data:
        username = data['username']
        message = data['message']
        sentiment = data['sentiment']

        # Prétraitement du message avec le stemming
        preprocessed_message = preprocess_text(message)

        # Vectorisation et transformation TF-IDF
        message_vect = count_vectorizer.transform([preprocessed_message])
        message_tfidf = tfidf_transformer.transform(message_vect)

        # Faire la prédiction avec le modèle de régression linéaire
        prediction = linear_model.predict(message_tfidf)
        predicted_sentiment = 1 if prediction[0] > 0.5 else 0

        # Comparer le sentiment prédit avec le sentiment fourni
        if predicted_sentiment == int(sentiment):
            response = "J'ai bien compris tes sentiments."
        else:
            response = "Désolé, je n'ai pas bien compris tes sentiments."

        return jsonify({'result': response}), 200
    else:
        return jsonify({'result': 'Données invalides'}), 400

@app.route('/exit')
def exit_app():
    return "Merci d'avoir utilisé l'application. L'application est maintenant fermée."

if __name__ == '__main__':
    app.run(debug=True)
