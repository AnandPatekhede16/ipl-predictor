import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import json

# Load dataset
df = pd.read_csv("data/IPL.csv", low_memory=False)

# Select columns
columns_needed = [
    'batting_team',
    'bowling_team',
    'venue',
    'city',
    'toss_winner',
    'toss_decision',
    'match_won_by'
]

df = df[columns_needed]
df = df.dropna()

print("Shape after cleaning:", df.shape)

# Encode columns separately
encoders = {}

for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save encoders
joblib.dump(encoders, "model/encoders.pkl")

# Split data
X = df.drop('match_won_by', axis=1)
y = df['match_won_by']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "model/ipl_model.pkl")

print("Model saved successfully!")

# Save team mapping (for output)
team_encoder = encoders['match_won_by']
team_mapping = dict(enumerate(team_encoder.classes_))

with open("model/team_mapping.json", "w") as f:
    json.dump(team_mapping, f)

print("Mapping saved successfully!")