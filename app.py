from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json

app = Flask(__name__)
CORS(app)

# Load model and encoders
model = joblib.load("model/ipl_model.pkl")
encoders = joblib.load("model/encoders.pkl")

# Load team mapping
with open("model/team_mapping.json") as f:
    team_mapping = json.load(f)

@app.route('/')
def home():
    return "IPL Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Encode inputs
    batting_team = encoders['batting_team'].transform([data['batting_team']])[0]
    bowling_team = encoders['bowling_team'].transform([data['bowling_team']])[0]
    venue = encoders['venue'].transform([data['venue']])[0]
    city = encoders['city'].transform([data['city']])[0]
    toss_winner = encoders['toss_winner'].transform([data['toss_winner']])[0]
    toss_decision = encoders['toss_decision'].transform([data['toss_decision']])[0]

    features = np.array([[batting_team, bowling_team, venue, city, toss_winner, toss_decision]])

    # Prediction
    probs = model.predict_proba(features)
    confidence = max(probs[0]) * 100

    prediction = model.predict(features)[0]
    team_name = team_mapping[str(prediction)]

    return jsonify({
        'prediction': team_name,
        'confidence': round(confidence, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)