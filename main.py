from flask import Flask, render_template

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

