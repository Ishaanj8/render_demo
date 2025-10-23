from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model
with open('trained_model.sav', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        values = [float(x) for x in request.form.values()]
        final_features = [np.array(values)]
        prediction = model.predict(final_features)

        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        return render_template('index.html', prediction_text=f'Patient is likely {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

import socket

if __name__ == '__main__':
    # Try to find a free port starting from 5000
    port = 5000
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                break  # Found a free port
            port += 1  # Try next port if busy

    print(f"🚀 Flask app running on port {port}")
    app.run(debug=True, port=port)