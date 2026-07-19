import streamlit as st
import joblib
from groq import Groq
from pathlib import Path
from config import GROQ_API_KEY

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="MindScan",
    page_icon="🧠",
    layout="centered"
)

# --------------------------
# LOAD MODEL
# --------------------------
model = joblib.load(MODEL_DIR / "mental_health_model.pkl")
vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")

# --------------------------
# GROQ CLIENT
# --------------------------
client = Groq(api_key=GROQ_API_KEY)

# --------------------------
# TRIGGER WORDS
# --------------------------
def get_trigger_words(text):

    trigger_words = [
        "stress",
        "stressed",
        "anxiety",
        "anxious",
        "depressed",
        "sad",
        "overwhelmed",
        "pressure",
        "exam",
        "deadline",
        "sleep",
        "lonely",
        "fear",
        "worried",
        "burnout",
        "tired",
        "exhausted",
        "panic",
        "failure",
        "hopeless"
    ]

    found = []

    text = text.lower()

    for word in trigger_words:
        if word in text:
            found.append(word)

    return found
def get_positive_words(text):

    positive_words = [
        "happy",
        "relaxed",
        "motivated",
        "confident",
        "productive",
        "good",
        "great",
        "excellent",
        "enjoy",
        "love",
        "best",
        "successful",
        "focused",
        "improving",
        "calm"
    ]

    found = []

    text = text.lower()

    for word in positive_words:
        if word in text:
            found.append(word)

    return found
# --------------------------
# TITLE
# --------------------------
st.title("🧠 MindScan")

st.subheader(
    "AI-Powered Student Wellness & Mental Health Assistant"
)

# --------------------------
# USER INPUTS
# --------------------------
sleep_hours = st.slider(
    "😴 Sleep Hours",
    0,
    12,
    7
)

study_hours = st.slider(
    "📚 Study Hours",
    0,
    15,
    4
)

screen_time = st.slider(
    "📱 Screen Time (Hours)",
    0,
    15,
    5
)
total_hours = sleep_hours + study_hours + screen_time
st.write(f"Total Tracked Hours: {total_hours}/24")
if total_hours > 24:
    st.error(
        "⚠ Total hours exceed 24. Please adjust your inputs."
    )
    st.stop()
user_text = st.text_area(
    "✍ Enter your journal entry",
    height=200,
    placeholder="Example: I feel overwhelmed by exams and deadlines..."
)

# --------------------------
# ANALYZE BUTTON
# --------------------------

if st.button("🔍 Analyze"):

    if user_text.strip() == "":
        st.warning("Please enter your journal entry.")
        st.stop()

    # --------------------------
    # PREDICTION
    # --------------------------

    X = vectorizer.transform([user_text])

    prediction = model.predict(X)[0]

    confidence = abs(
        model.decision_function(X)[0]
    )

    # --------------------------
    # WORD ANALYSIS
    # --------------------------

    trigger_words = get_trigger_words(
        user_text
    )

    positive_words = get_positive_words(
        user_text
    )

    positive_count = len(
        positive_words
    )
    trigger_count = len(trigger_words)
    # --------------------------
    # WELLNESS SCORE
    # --------------------------

    wellness_score = 100

    if sleep_hours < 6:
        wellness_score -= 20

    elif sleep_hours >= 7:
        wellness_score += 5

    if study_hours > 10:
        wellness_score -= 15

    elif 2 <= study_hours <= 8:
        wellness_score += 5

    if screen_time > 8:
        wellness_score -= 15

    if prediction == 1:
        wellness_score -= 20

    if positive_count >= 2:
        wellness_score += 10

    wellness_score = max(
        0,
        min(100, wellness_score)
    )

    # --------------------------
    # RISK SCORE CALCULATION
    # --------------------------

    risk_score = 0

    # ML Prediction

    if prediction == 1:
        risk_score += 30

    # Trigger Words

    risk_score += trigger_count * 10

    # Sleep

    if sleep_hours < 5:
        risk_score += 20

    elif sleep_hours < 7:
        risk_score += 10

    # Study Load

    if study_hours > 10:
        risk_score += 15

    elif study_hours > 8:
        risk_score += 10

    # Screen Time

    if screen_time > 10:
        risk_score += 15

    elif screen_time > 8:
        risk_score += 10

    # Positive Words

    risk_score -= positive_count * 5

    risk_score = max(
        0,
        min(100, risk_score)
    )

    # --------------------------
    # FINAL RISK LEVEL
    # --------------------------

    if risk_score >= 50:
        risk_level = "High"

    elif risk_score >= 20:
        risk_level = "Moderate"

    else:
        risk_level = "Low"

    # --------------------------
    # DISPLAY RESULTS
    # --------------------------

    st.markdown("---")

    st.header(
        "📊 Assessment Result"
    )

    if risk_level == "High":

        st.error(
            "🔴 High Wellness Risk"
        )

    elif risk_level == "Moderate":

        st.warning(
            "🟡 Moderate Wellness Risk"
        )

    else:

        st.success(
            "🟢 Low Wellness Risk"
        )

    st.write(
        f"**Confidence Score:** {confidence:.2f}"
    )

    st.write(
        f"**Risk Score:** {risk_score}/100"
    )

    # --------------------------
    # WELLNESS SCORE
    # --------------------------

    st.subheader(
        "💚 Wellness Score"
    )

    st.progress(
        wellness_score / 100
    )

    st.write(
        f"Overall Wellness Score: **{wellness_score}/100**"
    )

    # --------------------------
    # TRIGGER WORDS
    # --------------------------

    if trigger_words:

        st.subheader(
            "🚨 Trigger Words Detected"
        )

        for word in trigger_words:

            st.write(
                f"• {word}"
            )

    # --------------------------
    # POSITIVE WORDS
    # --------------------------

    if positive_words:

        st.subheader(
            "😊 Positive Indicators"
        )

        for word in positive_words:

            st.write(
                f"• {word}"
            )

    # --------------------------
    # ISSUES
    # --------------------------

    issues = []

    if sleep_hours < 6:
        issues.append(
            "Low Sleep"
        )

    if study_hours > 8:
        issues.append(
            "High Study Load"
        )

    if screen_time > 8:
        issues.append(
            "High Screen Time"
        )

    if risk_level == "High":
        issues.append(
            "Mental Wellness Risk"
        )

    issues_text = ", ".join(
        issues
    )

    # --------------------------
    # AI PROMPT
    # --------------------------

    prompt = f"""
You are MindScan AI.

You are an expert student wellness coach,
study mentor,
productivity coach,
and motivational guide.

Student Journal:
{user_text}

Sleep Hours:
{sleep_hours}

Study Hours:
{study_hours}

Screen Time:
{screen_time}

Risk Level:
{risk_level}

Wellness Score:
{wellness_score}

Detected Issues:
{issues_text}

Positive Indicators:
{', '.join(positive_words)}

Trigger Words:
{', '.join(trigger_words)}

Provide:

1. Mental Health Analysis

2. Wellness Recommendations
(At least 5)

3. Study Improvement Tips
(At least 5)

4. Sleep Improvement Advice

5. Screen Time Management

6. Motivation Corner

7. Daily Action Plan

Use bullet points.

Be encouraging.

Do not diagnose diseases.

Keep response around 300 words.
"""

    # --------------------------
    # AI RESPONSE
    # --------------------------

    with st.spinner(
        "🤖 Generating personalized recommendations..."
    ):

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        advice = (
            response
            .choices[0]
            .message
            .content
        )

    st.markdown("---")

    st.header(
        "🤖 AI Wellness Coach"
    )

    st.write(
        advice
    )