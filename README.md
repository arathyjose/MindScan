# MindScan – Student Stress & Burnout Early Detection System

MindScan is a machine learning-based tool that analyzes written text, such as journal entries, to help identify early signs of stress and burnout in students. It predicts a mental health risk category from the text and offers personalized wellness recommendations through a simple web interface.

## What It Does

- Takes a piece of written text (e.g. a journal entry) as input
- Classifies the text into a stress/burnout risk category using a trained machine learning model
- Displays the predicted risk level along with personalized wellness recommendations
- Runs as an interactive web app built with Streamlit

## How It Works

The core of MindScan is a text classification pipeline:

1. **Data Preprocessing** – Cleans and prepares raw text data for analysis
2. **Feature Extraction** – Converts text into numerical features using **TF-IDF vectorization**
3. **Model** – A **Linear Support Vector Classifier (LinearSVC)** trained on the processed data
4. **Evaluation** – Achieved approximately **89% classification accuracy** on the test data
5. **Deployment** – The trained model is integrated into a **Streamlit** app for real-time predictions

## Tech Stack

- **Language:** Python
- **ML/Data:** Scikit-learn, Pandas, TF-IDF, LinearSVC
- **App Framework:** Streamlit
- **Model Persistence:** Joblib

## Project Structure

```
MindScan/
├── app.py              # Streamlit web application
├── train.py             # Model training script
├── predict.py            # Prediction logic
├── config.py             # Configuration (API keys loaded via .env)
├── data/                # Datasets used for training
├── models/               # Saved trained model and vectorizer
└── requirements.txt         # Project dependencies
```

## Screenshots

[App Home](screenshots/home.png)
[Prediction Result](screenshots/result.png)
-->

## Motivation

Student stress and burnout often go unnoticed until they become serious. MindScan was built to explore how simple, interpretable machine learning models can be used to flag early warning signs from everyday writing, making it easier to encourage timely support and self-awareness.

## Author

**Arathy Jose**
[GitHub](https://github.com/arathyjose) | [LinkedIn](https://linkedin.com/in/arathy2002)