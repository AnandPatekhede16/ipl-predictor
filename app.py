from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json
import os

app = Flask(__name__)
CORS(app)

# -------------------------------
# Load model and encoders safely
# -------------------------------
try:
    model = joblib.load("model/ipl_model.pkl")
    encoders = joblib.load("model/encoders.pkl")

    with open("model/team_mapping.json") as f:
        team_mapping = json.load(f)

    print("✅ Model and files loaded successfully")

except Exception as e:
    print("❌ Error loading model files:", e)
    model = None
    encoders = None
    team_mapping = {}

# -------------------------------
# Home route
# -------------------------------
@app.route('/')
def home():
    return "🏏 IPL Prediction API is running!"

# -------------------------------
# Health check (for deployment)
# -------------------------------
@app.route('/health')
def health():
    return jsonify({"status": "ok"})

# -------------------------------
# Prediction route
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.json

        # Encode inputs safely
        batting_team = encoders['batting_team'].transform([data['batting_team']])[0]
        bowling_team = encoders['bowling_team'].transform([data['bowling_team']])[0]
        venue = encoders['venue'].transform([data['venue']])[0]
        city = encoders['city'].transform([data['city']])[0]
        toss_winner = encoders['toss_winner'].transform([data['toss_winner']])[0]
        toss_decision = encoders['toss_decision'].transform([data['toss_decision']])[0]

        features = np.array([[batting_team, bowling_team, venue, city, toss_winner, toss_decision]])

        # Prediction
        probs = model.predict_proba(features)
        confidence = float(max(probs[0]) * 100)

        prediction = model.predict(features)[0]
        team_name = team_mapping.get(str(prediction), "Unknown Team")

        return jsonify({
            'prediction': team_name,
            'confidence': round(confidence, 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400

# -------------------------------
# For Gunicorn (IMPORTANT)
# -------------------------------
# This line is REQUIRED for Render
application = app

# -------------------------------
# Run app (for local only)
# -------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)