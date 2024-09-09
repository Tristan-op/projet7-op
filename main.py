import os
import subprocess
import sys

# Vérifier si pip est installé et essayer de l'installer si nécessaire
try:
    import pip
except ImportError:
    print("pip non installé, installation de pip...")
    subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])

# Installer les packages du fichier requirements.txt
def install_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
        except subprocess.CalledProcessError as e:
            print(f"Erreur lors de l'installation des dépendances: {e}")
    else:
        print("Le fichier requirements.txt est introuvable.")

# Appeler la fonction pour installer les dépendances
install_requirements()

# Ensuite, importer les bibliothèques après l'installation

# Le reste de ton code Flask ou autre
from flask import Flask, jsonify

app = Flask(__name__)

# Route pour la page Welcome
@app.route('/')
def welcome():
    return render_template('welcome.html')
