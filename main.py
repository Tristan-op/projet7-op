from flask import Flask, render_template

app = Flask(__name__)

# Route pour la page Welcome
@app.route('/')
def welcome():
    return render_template('welcome.html')

if __name__ == '__main__':
    app.run(debug=True)
