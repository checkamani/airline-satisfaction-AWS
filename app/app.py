
from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# load model
pipe = joblib.load("model/model.pkl")

@app.route("/")
def home():
    return "<h2>Airline Satisfaction Model is Running</h2>"

@app.route("/health")
def health():
    return "ok", 200

@app.route("/predict")
def predict():
    df = pd.read_csv("train_full.csv")
    X_sample = df.drop(columns=["satisfaction"]).iloc[[0]].copy()

    pred = pipe.predict(X_sample)[0]
    proba = pipe.predict_proba(X_sample)[0,1]

    return {
        "prediction": int(pred),
        "probability_satisfied": float(proba)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
