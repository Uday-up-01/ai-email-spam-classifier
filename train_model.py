"""
Milestone 3: Machine Learning Model Development
AI-Based Email Spam Classification System
"""

import pickle
import json

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model, X_test, y_test, model_name):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    cm = confusion_matrix(y_test, predictions)

    print("\n" + "=" * 55)
    print(f"MODEL: {model_name}")
    print("=" * 55)

    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Ham", "Spam"],
            zero_division=0
        )
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist()
    }


def main():

    print("\nAI-BASED EMAIL SPAM CLASSIFICATION")
    print("MACHINE LEARNING MODEL TRAINING")
    print("=" * 55)

    # Load processed dataset
    with open("processed_data.pkl", "rb") as file:
        X_train, X_test, y_train, y_test = pickle.load(file)

    print("\nDataset loaded successfully.")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples : {X_test.shape[0]}")

    # Model 1: Multinomial Naive Bayes
    print("\nTraining Multinomial Naive Bayes...")

    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    nb_results = evaluate_model(
        nb_model,
        X_test,
        y_test,
        "Multinomial Naive Bayes"
    )

    # Model 2: Logistic Regression
    print("\nTraining Logistic Regression...")

    lr_model = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    lr_model.fit(X_train, y_train)

    lr_results = evaluate_model(
        lr_model,
        X_test,
        y_test,
        "Logistic Regression"
    )

    # Compare models
    print("\n" + "=" * 55)
    print("MODEL COMPARISON")
    print("=" * 55)

    print(f"{'Metric':<15}{'Naive Bayes':<18}{'Logistic Regression'}")

    for metric in ["accuracy", "precision", "recall", "f1_score"]:

        print(
            f"{metric:<15}"
            f"{nb_results[metric] * 100:<18.2f}"
            f"{lr_results[metric] * 100:.2f}%"
        )

    # Select best model based on F1 score
    if nb_results["f1_score"] >= lr_results["f1_score"]:
        best_model = nb_model
        best_name = "Multinomial Naive Bayes"
        best_results = nb_results
    else:
        best_model = lr_model
        best_name = "Logistic Regression"
        best_results = lr_results

    print("\nSelected Model:", best_name)
    print(f"Best F1 Score: {best_results['f1_score'] * 100:.2f}%")

    # Save selected model
    with open("spam_classifier_model.pkl", "wb") as file:
        pickle.dump(best_model, file)

    # Save comparison results
    comparison = {
        "naive_bayes": nb_results,
        "logistic_regression": lr_results,
        "selected_model": best_name,
        "best_results": best_results
    }

    with open("model_comparison.json", "w") as file:
        json.dump(comparison, file, indent=4)

    print("\n" + "=" * 55)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 55)

    print("Saved: spam_classifier_model.pkl")
    print("Saved: model_comparison.json")


if __name__ == "__main__":
    main()