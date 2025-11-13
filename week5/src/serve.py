"""
Model Serving Script
--------------------
Deploys trained model as an API endpoint using Flask.
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("../models/readmission_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
