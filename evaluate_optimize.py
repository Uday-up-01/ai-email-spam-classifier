"""
Milestone 4: Model Evaluation and Optimization
AI-Based Email Spam Classification System
"""

import pickle
import json

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model, X_test, y_test, name):

    predictions = model.predict(X_test)

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    results = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "f1_score": float(f1_score(y_test, predictions))
    }

    for metric, value in results.items():
        print(f"{metric:<12}: {value * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Ham", "Spam"]
    ))

    return results


def main():

    print("\nMODEL EVALUATION AND OPTIMIZATION")
    print("=" * 55)

    # Load processed dataset
    with open("processed_data.pkl", "rb") as file:
        X_train, X_test, y_train, y_test = pickle.load(file)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # 1. Cross-validation
    print("\n1. FIVE-FOLD CROSS VALIDATION")

    nb = MultinomialNB()

    lr = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    models = {
        "Naive Bayes": nb,
        "Logistic Regression": lr
    }

    cv_results = {}

    for name, model in models.items():

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1",
            n_jobs=-1
        )

        cv_results[name] = {
            "mean_f1": float(scores.mean()),
            "std_f1": float(scores.std())
        }

        print(f"\n{name}")
        print("Fold scores:", scores)
        print(f"Mean F1: {scores.mean():.4f}")
        print(f"Standard deviation: {scores.std():.4f}")

    # 2. Hyperparameter tuning
    print("\n2. GRID SEARCH OPTIMIZATION")

    param_grid = {
        "alpha": [0.1, 0.5, 1.0, 2.0, 5.0]
    }

    nb_search = GridSearchCV(
        MultinomialNB(),
        param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1
    )

    nb_search.fit(X_train, y_train)

    print("\nNaive Bayes best parameters:")
    print(nb_search.best_params_)

    print("Best CV F1:", nb_search.best_score_)

    lr_grid = {
        "C": [0.1, 1, 10, 100]
    }

    lr_search = GridSearchCV(
        LogisticRegression(
            max_iter=2000,
            random_state=42
        ),
        lr_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1
    )

    lr_search.fit(X_train, y_train)

    print("\nLogistic Regression best parameters:")
    print(lr_search.best_params_)

    print("Best CV F1:", lr_search.best_score_)

    # 3. Select optimized model
    candidates = {
        "Naive Bayes": nb_search.best_estimator_,
        "Logistic Regression": lr_search.best_estimator_
    }

    best_name = max(
        candidates,
        key=lambda name: (
            nb_search.best_score_
            if name == "Naive Bayes"
            else lr_search.best_score_
        )
    )

    best_model = candidates[best_name]

    # 4. Final evaluation
    print("\n3. FINAL TEST SET EVALUATION")

    final_results = evaluate_model(
        best_model,
        X_test,
        y_test,
        best_name
    )

    # 5. Save optimized model
    with open("optimized_spam_classifier.pkl", "wb") as file:
        pickle.dump(best_model, file)

    report = {
        "cross_validation": cv_results,
        "naive_bayes_best_params": nb_search.best_params_,
        "logistic_regression_best_params": lr_search.best_params_,
        "selected_model": best_name,
        "final_test_results": final_results
    }

    with open("optimization_results.json", "w") as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 55)
    print("OPTIMIZATION COMPLETED")
    print("=" * 55)

    print("Selected model:", best_name)
    print("Saved: optimized_spam_classifier.pkl")
    print("Saved: optimization_results.json")


if __name__ == "__main__":
    main()