import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow import keras
import joblib

# Load Model & Scaler
def load_model(model_path="models/model.keras", scaler_path="models/scaler.pkl"):
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        print("Model loaded successfully.")
        return model, scaler
    else:
        print("Model files not found! Train the model first.")
        return None, None

# Initialize Flask App
app = Flask(__name__)
model, scaler = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = np.array(data['features']).reshape(1, -1)
        
        print("Number of Features in API Request:", input_data.shape[1])  

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0][0]

        return jsonify({'DON_Concentration': float(prediction)})
    
    except Exception as e:
        print("API Error:", str(e))
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    if model is not None and scaler is not None:
        print("Starting API Server...")
        app.run(host='0.0.0.0', port=2222)
    else:
        print("API cannot start. Train the model first using `python3 train.py`.")
