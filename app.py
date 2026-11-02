import os
import re
import csv
import joblib
import numpy as np

from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Initialize Flask application
app = Flask(__name__)
app.secret_key = "spam-classifier-project-key"


# Define project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "optimized_spam_classifier.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# Load trained model and TF-IDF vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

HISTORY_FILE = os.path.join(BASE_DIR, "prediction_history.csv")


# Preserve negative words during preprocessing
STOP_WORDS = set(ENGLISH_STOP_WORDS) - {"no", "not", "nor"}


def clean_text(text):
    """
    Apply the same text cleaning used during model training.
    """

    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove punctuation, numbers and special characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove stopwords and words shorter than 3 characters
    tokens = [
        word
        for word in text.split()
        if len(word) > 2 and word not in STOP_WORDS
    ]

    return " ".join(tokens)


def save_history(email, prediction, confidence):

    file_exists = os.path.isfile(HISTORY_FILE)

    with open(
        HISTORY_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Date",
                "Email",
                "Prediction",
                "Confidence"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            email,
            prediction,
            f"{confidence:.2f}%"
        ])


def classify_email(email):

    cleaned_email = clean_text(email)

    if not cleaned_email:
        raise ValueError("Please enter a valid email.")

    # Convert email into TF-IDF features
    email_vector = vectorizer.transform([cleaned_email])

    # Predict email category
    prediction = model.predict(email_vector)[0]

    # Calculate prediction confidence
    probabilities = model.predict_proba(email_vector)[0]
    confidence = float(np.max(probabilities)) * 100

    # Determine classification label
    if prediction == 1:
        label = "SPAM"
    else:
        label = "NOT SPAM"

    # Identify words contributing towards spam
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    present_features = email_vector.toarray()[0]

    word_scores = []

    for index, value in enumerate(present_features):

        if value > 0 and coefficients[index] > 0:

            word_scores.append({
                "word": feature_names[index],
                "score": float(value * coefficients[index])
            })

    # Sort words by spam contribution
    word_scores.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # Select top five spam indicator words
    spam_indicators = [
        item["word"]
        for item in word_scores[:5]
    ]

    return label, confidence, spam_indicators


@app.route("/", methods=["GET", "POST"])
def index():

    result = None

    if request.method == "POST":

        email_text = request.form.get("email", "").strip()

        if not email_text:
            flash("Please enter an email before checking.")

        else:

            try:

                label, confidence, spam_indicators = classify_email(email_text)

                result = {
                    "email": email_text,
                    "prediction": label,
                    "confidence": round(confidence, 2),
                    "spam_indicators": spam_indicators
                }

                save_history(email_text, label, confidence)

            except Exception as error:
                flash(f"Prediction error: {error}")

    return render_template(
        "index.html",
        result=result,
        batch_results=None
    )


@app.route("/batch", methods=["POST"])
def batch_classify():

    uploaded_file = request.files.get("file")

    if not uploaded_file or uploaded_file.filename == "":
        flash("Please select a text file.")
        return redirect(url_for("index"))

    if not uploaded_file.filename.lower().endswith(".txt"):
        flash("Only .txt files are supported.")
        return redirect(url_for("index"))

    try:

        content = uploaded_file.read().decode("utf-8-sig")

        # Each non-empty line represents one email
        emails = [
            line.strip()
            for line in content.splitlines()
            if line.strip()
        ]

        if not emails:
            flash("The uploaded file is empty.")
            return redirect(url_for("index"))

        results = []

        for email in emails:

            label, confidence, spam_indicators = classify_email(email)

            results.append({
                "email": email,
                "prediction": label,
                "confidence": round(confidence, 2),
                "spam_indicators": spam_indicators
            })

            save_history(email, label, confidence)

        return render_template(
            "index.html",
            result=None,
            batch_results=results
        )

    except UnicodeDecodeError:
        flash("Unable to read the file. Please upload a UTF-8 text file.")

    except Exception as error:
        flash(f"Batch processing error: {error}")

    return redirect(url_for("index"))


@app.route("/history")
def history():

    records = []

    if os.path.isfile(HISTORY_FILE):

        with open(
            HISTORY_FILE,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)
            records = list(reader)

    return render_template(
        "history.html",
        records=records
    )

@app.route("/dashboard")
def dashboard():

    records = []

    total_emails = 0
    spam_count = 0
    ham_count = 0
    spam_percentage = 0

    # Read prediction history
    if os.path.isfile(HISTORY_FILE):

        try:
            with open(
                HISTORY_FILE,
                "r",
                newline="",
                encoding="utf-8"
            ) as file:

                reader = csv.DictReader(file)
                records = list(reader)

            total_emails = len(records)

            spam_count = sum(
                1 for record in records
                if record.get("Prediction") == "SPAM"
            )

            ham_count = sum(
                1 for record in records
                if record.get("Prediction") == "NOT SPAM"
            )

            if total_emails > 0:
                spam_percentage = round(
                    (spam_count / total_emails) * 100, 2
                )

        except Exception as error:
            print("Dashboard history error:", error)

    # Prepare last 7 days of analytics
    today = datetime.now().date()

    daily_labels = []
    daily_total = []
    daily_spam = []
    daily_ham = []

    for i in range(6, -1, -1):

        current_date = today - timedelta(days=i)

        daily_labels.append(current_date.strftime("%d %b"))

        day_records = []

        for record in records:

            try:
                record_date = datetime.strptime(
                    record.get("Date", ""),
                    "%Y-%m-%d %H:%M:%S"
                ).date()

                if record_date == current_date:
                    day_records.append(record)

            except (ValueError, TypeError):
                continue

        day_spam = sum(
            1 for record in day_records
            if record.get("Prediction") == "SPAM"
        )

        day_ham = sum(
            1 for record in day_records
            if record.get("Prediction") == "NOT SPAM"
        )

        daily_total.append(len(day_records))
        daily_spam.append(day_spam)
        daily_ham.append(day_ham)

    # Latest five predictions
    recent_records = records[-5:][::-1]

    return render_template(
        "dashboard.html",

        total_emails=total_emails,
        spam_count=spam_count,
        ham_count=ham_count,
        spam_percentage=spam_percentage,

        recent_records=recent_records,

        daily_labels=daily_labels,
        daily_total=daily_total,
        daily_spam=daily_spam,
        daily_ham=daily_ham
    )

if __name__ == "__main__":

    print("Starting AI Email Spam Classification System...")

    app.run(debug=True)