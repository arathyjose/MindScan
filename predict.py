import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Load saved model and vectorizer
model = joblib.load(MODEL_DIR / "mental_health_model.pkl")
vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")

# Get user input
text = input("Enter Journal Text: ")

# Convert text into TF-IDF features
X = vectorizer.transform([text])

# Predict
prediction = model.predict(X)[0]

print("\n======================")

if prediction == 1:
    print("⚠ Mental Health Risk Detected")
else:
    print("✅ Low Risk")

print("======================")