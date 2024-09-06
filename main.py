from flask import Flask, request, jsonify
import sqlite3
from datetime import datetime

app = Flask(__name__)

# Création de la base de données
def init_db():
    with sqlite3.connect('chat.db') as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                message TEXT NOT NULL,
                sentiment INTEGER NOT NULL,
                time TEXT NOT NULL
            )
        ''')
    print("Base de données initialisée.")

# Ajouter un message dans la base de données
def add_message(username, message, sentiment):
    with sqlite3.connect('chat.db') as conn:
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO messages (username, message, sentiment, time)
            VALUES (?, ?, ?, ?)
        ''', (username, message, sentiment, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()

# Récupérer l'historique des messages
def get_chat_history():
    with sqlite3.connect('chat.db') as conn:
        cur = conn.cursor()
        cur.execute('SELECT username, message, sentiment, time FROM messages ORDER BY time DESC')
        return cur.fetchall()

@app.route('/send-message', methods=['POST'])
def send_message():
    data = request.get_json()
    username = data['username']
    message = data['message']
    sentiment = data['sentiment']

    add_message(username, message, sentiment)

    return jsonify({'message': 'Message envoyé avec succès'})

@app.route('/chat-history', methods=['GET'])
def chat_history():
    messages = get_chat_history()
    return jsonify({
        'messages': [
            {'username': msg[0], 'message': msg[1], 'sentiment': msg[2], 'time': msg[3]}
        for msg in messages]
    })

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
