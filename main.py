from flask import Flask, render_template, redirect, url_for

# Créer l'instance Flask
app = Flask(__name__, template_folder="templates")

# Page d'accueil (welcome.html)
@app.route('/')
def home():
    """
    Cette fonction affiche la page d'accueil 'welcome.html'
    """
    return render_template('welcome.html')

# Page de chat (chat.html)
@app.route('/continue')
def continue_to_chat():
    """
    Cette fonction redirige l'utilisateur vers la page de chat 'chat.html'
    """
    return render_template('chat.html')

# Fonction pour quitter l'API
@app.route('/exit')
def exit_app():
    """
    Cette fonction quitte l'application Flask.
    """
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Le serveur ne peut pas être arrêté.')
    func()
    return "Application arrêtée. Vous pouvez fermer cette fenêtre."
