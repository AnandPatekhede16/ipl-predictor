from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import json
import os
import gdown

app = Flask(__name__)
CORS(app)

# -------------------------------
# Google Drive URLs
# -------------------------------
MODEL_URL = "https://drive.google.com/uc?id=1rN8za-D_gfZ4GIu6hJe4Q91zkWC9PoKN"
ENCODER_URL = "https://drive.google.com/uc?id=1yOZ_P6KPzycxUfvcsJWDmOpm1CUv1R_C"
JSON_URL = "https://drive.google.com/uc?id=1qDXpqt1qltKF2dIwmDxgMxLMiajxi1GK"

os.makedirs("model", exist_ok=True)

# -------------------------------
# Download helper
# -------------------------------
def download(url, path):
    if not os.path.exists(path):
        print(f"⬇️ Downloading {path}...")
        try:
            gdown.download(url, path, quiet=False)
        except Exception as e:
            print(f"❌ Failed to download {path}: {e}")

# Download files
download(MODEL_URL, "model/ipl_model.pkl")
download(ENCODER_URL, "model/encoders.pkl")
download(JSON_URL, "model/team_mapping.json")

# -------------------------------
# Load files safely
# -------------------------------
model = None
encoders = None
team_mapping = {}

try:
    print("📦 Loading model files...")

    if not os.path.exists("model/ipl_model.pkl"):
        print("❌ Model file missing")
    if not os.path.exists("model/encoders.pkl"):
        print("❌ Encoder file missing")
    if not os.path.exists("model/team_mapping.json"):
        print("❌ JSON mapping file missing")

    model = joblib.load("model/ipl_model.pkl")
    encoders = joblib.load("model/encoders.pkl")

    with open("model/team_mapping.json") as f:
        team_mapping = json.load(f)

    print("✅ Model and files loaded successfully")

except Exception as e:
    print("❌ Error loading model files:", e)

# -------------------------------
# ROUTES
# -------------------------------
@app.route('/')
def home():
    return render_template("index.html")

# ✅ NEW: Health check (important for Render)
@app.route('/health')
def health():
    return jsonify({"status": "ok"})

@app.route('/options')
def options():
    try:
        return jsonify({
            "teams": list(encoders['batting_team'].classes_),
            "venues": list(encoders['venue'].classes_),
            "cities": list(encoders['city'].classes_)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or encoders is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.json

        features = np.array([[
            encoders['batting_team'].transform([data['batting_team']])[0],
            encoders['bowling_team'].transform([data['bowling_team']])[0],
            encoders['venue'].transform([data['venue']])[0],
            encoders['city'].transform([data['city']])[0],
            encoders['toss_winner'].transform([data['toss_winner']])[0],
            encoders['toss_decision'].transform([data['toss_decision']])[0]
        ]])

        probs = model.predict_proba(features)
        confidence = float(max(probs[0]) * 100)

        pred = model.predict(features)[0]
        team = team_mapping.get(str(pred), "Unknown")

        return jsonify({
            "prediction": team,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({"error": str(e)}), 400

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    print("🚀 Starting Flask app...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)