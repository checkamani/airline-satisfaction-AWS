from flask import Flask, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model (must exist in model/model.pkl)
pipe = joblib.load("model/model.pkl")


@app.route("/", methods=["GET"])
def home():
    return "<h2>Airline Satisfaction Model is Running</h2>"


@app.route("/health", methods=["GET"])
def health():
    return "ok", 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    X_sample = pd.DataFrame([data])

    pred = pipe.predict(X_sample)[0]
    proba = pipe.predict_proba(X_sample)[0, 1] if hasattr(pipe, "predict_proba") else None

    return {
        "prediction": int(pred),
        "probability_satisfied": float(proba) if proba is not None else None
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
