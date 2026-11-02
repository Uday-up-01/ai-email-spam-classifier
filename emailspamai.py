"""Milestone 2 (revised): leakage-safe email preprocessing.
Run from the folder containing spamassassin_sample.csv:
    python emailspamai.py
"""
import json
import pickle
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "spamassassin_sample.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_FEATURES = 3000

# Keep negation words because removing them can reverse an email's meaning.
STOP_WORDS = set(ENGLISH_STOP_WORDS) - {"no", "not", "nor"}


def clean_text(text: object) -> str:
    """Normalize an email body while retaining meaningful alphabetic tokens."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)                 # HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text) # URLs
    text = re.sub(r"[^a-z\s]", " ", text)               # punctuation/digits
    tokens = [w for w in text.split() if len(w) > 2 and w not in STOP_WORDS]
    return " ".join(tokens)


def main() -> None:
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    df = pd.read_csv(DATASET)
    required = {"body", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")

    df = df[["body", "label"]].dropna(subset=["body", "label"]).copy()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    df = df[df["label"].isin([0, 1])].copy()
    df["clean_body"] = df["body"].map(clean_text)

    X_text = df["clean_body"]
    y = df["label"]
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Fit the vectorizer ONLY on training data to avoid test-set vocabulary/IDF leakage.
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    with (BASE_DIR / "processed_data.pkl").open("wb") as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)
    with (BASE_DIR / "vectorizer.pkl").open("wb") as f:
        pickle.dump(vectorizer, f)

    # Raw cleaned text is retained for Pipeline-based CV, where TF-IDF is refit
    # inside each fold (avoids leakage between CV training and validation folds).
    text_data = {
        "X_train_text": X_train_text.reset_index(drop=True),
        "X_test_text": X_test_text.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
    }
    with (BASE_DIR / "text_data.pkl").open("wb") as f:
        pickle.dump(text_data, f)

    counts = df["label"].value_counts().to_dict()
    info = {
        "total_emails_after_cleaning": int(len(df)),
        "ham": int(counts.get(0, 0)),
        "spam": int(counts.get(1, 0)),
        "train_size": int(len(X_train_text)),
        "test_size": int(len(X_test_text)),
        "tfidf_features": int(X_train.shape[1]),
        "test_size_fraction": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "note": "TF-IDF fitted on training data only.",
    }
    (BASE_DIR / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print("\nPREPROCESSING COMPLETED")
    print(f"Emails retained : {len(df):,}")
    print(f"Ham             : {counts.get(0, 0):,}")
    print(f"Spam            : {counts.get(1, 0):,}")
    print(f"Training emails : {X_train.shape[0]:,}")
    print(f"Testing emails  : {X_test.shape[0]:,}")
    print(f"TF-IDF features : {X_train.shape[1]:,}")
    print("Saved: processed_data.pkl, vectorizer.pkl, text_data.pkl, dataset_info.json")


if __name__ == "__main__":
    main()
