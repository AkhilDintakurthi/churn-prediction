from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

# Load the trained pipeline
pipeline_path = os.path.join("model", "churn_pipeline.pkl")
pipeline = joblib.load(pipeline_path)

app = Flask(__name__)

@app.route('/')
def home():
    return "Churn Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        input_df = pd.DataFrame([input_data])

        pred = pipeline.predict(input_df)[0]
        prob = pipeline.predict_proba(input_df)[0][1]

        return jsonify({
            'prediction': 'Churn' if pred == 1 else 'No Churn',
            'probability': round(float(prob), 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

