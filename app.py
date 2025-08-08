from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le_mood = pickle.load(open("label_encoder.pkl", "rb"))
le_key = pickle.load(open("key_encoder.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    tempo = float(request.form['tempo'])
    bpm = float(request.form['bpm'])
    energy = float(request.form['energy'])
    loudness = float(request.form['loudness'])
    key = request.form['key']

    key_encoded = le_key.transform([key])[0]
    data = scaler.transform([[tempo, bpm, energy, loudness, key_encoded]])
    prediction = model.predict(data)
    mood = le_mood.inverse_transform(prediction)[0]

    return render_template('index.html', prediction_text=f"Predicted Mood: {mood}")

if __name__ == "__main__":
    app.run(debug=True)
