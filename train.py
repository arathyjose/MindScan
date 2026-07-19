import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
# ---------------------------------
# Load Dataset
# ---------------------------------
df = pd.read_csv("data/Mental-Health-Twitter.csv")

# Features and Labels
X_text = df["post_text"]
y = df["label"]

# ---------------------------------
# TF-IDF Vectorization
# ---------------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

X = vectorizer.fit_transform(X_text)

# ---------------------------------
# Train-Test Split
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------
# Train Model
# ---------------------------------
model = LinearSVC()
# ---------------------------------
# K-Fold Cross Validation
# ---------------------------------

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=skf,
    scoring="accuracy"
)

print("\n========== K-FOLD CROSS VALIDATION ==========\n")

print("Fold Accuracies:")
print(cv_scores)

print(
    f"\nMean CV Accuracy: {cv_scores.mean():.4f}"
)

print(
    f"Standard Deviation: {cv_scores.std():.4f}"
)
model.fit(X_train, y_train)

# ---------------------------------
# Predictions
# ---------------------------------
predictions = model.predict(X_test)
roc_auc = roc_auc_score(
    y_test,
    predictions
)
# ---------------------------------
# Evaluation Metrics
# ---------------------------------
accuracy = accuracy_score(y_test, predictions)

precision = precision_score(y_test, predictions)

recall = recall_score(y_test, predictions)

f1 = f1_score(y_test, predictions)

cm = confusion_matrix(y_test, predictions)

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)

sensitivity = tp / (tp + fn)
balanced_acc = balanced_accuracy_score(
    y_test,
    predictions
)


# ---------------------------------
# Results
# ---------------------------------
print("\n========== MODEL EVALUATION ==========\n")

print(f"Accuracy    : {accuracy:.4f}")
print(f"Balanced Accuracy : {balanced_acc:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"F1 Score    : {f1:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"Sensitivity : {sensitivity:.4f}")
print(f"ROC-AUC     : {roc_auc:.4f}")

print("\nConfusion Matrix:\n")
print(cm)

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))
# ---------------------------------
# Save Model
# ---------------------------------
joblib.dump(
    model,
    "models/mental_health_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/vectorizer.pkl"
)

print("\nModel Saved Successfully!")