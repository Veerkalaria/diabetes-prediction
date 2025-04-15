from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load models and preprocessor
model_paths = {
    "Logistic_Regression": joblib.load("saved_models/Logistic_Regression.joblib"),
    "Gradient_Boosting": joblib.load("saved_models/Gradient_Boosting.joblib"),
    "Random_Forest": joblib.load("saved_models/Random_Forest.joblib")
}
preprocessor = joblib.load("saved_models/preprocessor.joblib")
label_encoder = joblib.load("saved_models/label_encoder.joblib")

# Choose a default model
model = model_paths["Random_Forest"]

@app.route('/')
def home():
    return "Diabetes Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input JSON
        input_data = request.get_json()
        df = pd.DataFrame([input_data])

        # Preprocess
        transformed_input = preprocessor.transform(df)

        # Predict
        prediction = model.predict(transformed_input)
        prediction_label = label_encoder.inverse_transform(prediction)[0]

        return jsonify({
            'prediction': int(prediction[0]),
            'label': prediction_label
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
