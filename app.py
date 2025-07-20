
from flask import Flask, request, render_template # type: ignore
import pickle
import numpy as np # type: ignore
import pandas as pd  # type: ignore # Optional if using DataFrame

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract values using updated form names
    hco3 = int(request.form['hco3'])
    so4 = int(request.form['so4'])
    mg = int(request.form['mg'])
    sr = int(request.form['sr'])

    features = [hco3, so4, mg, sr]
    final_features = [np.array(features)]

    prediction = model.predict(final_features)
    output = 'No breakthrough' if prediction[0] == 1 else 'Breakthrough happened'

    return render_template('index.html', prediction_text=f'Prediction: {output}')


if __name__ == "__main__":
    app.run(debug=True)

