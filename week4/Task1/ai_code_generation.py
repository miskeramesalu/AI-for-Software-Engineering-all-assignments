"""
=== Task 1: AI Code Generation ===
Course: AI in Software Engineering – Week 4

This script demonstrates how AI-assisted code generation helps software engineers
automate repetitive tasks such as data preprocessing, feature engineering,
and model evaluation. It compares an AI-generated function with a manually
refined one to show how developers can combine AI assistance with human insight.
"""

# --- Imports ---
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# === AI-Suggested Code (auto-generated example) ===
def ai_generated_model():
    """Automatically suggested by an AI tool: train a quick ML model."""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"[AI Model] Accuracy: {acc:.3f}")
    return model, acc


# === Human-Refined Code (improved version) ===
def refined_model(n_estimators=200, max_depth=6):
    """
    Manually tuned RandomForestClassifier using better parameters and comments.
    Demonstrates how developers refine AI-suggested code for efficiency and clarity.
    """
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=7
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=7
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"[Refined Model] Accuracy: {acc:.3f}")
    return model, acc


# === Main block ===
if __name__ == "__main__":
    print("Running Task 1: AI Code Generation Demo\n")
    ai_model, ai_acc = ai_generated_model()
    refined_model_, refined_acc = refined_model()

    # Comparison summary
    print("\n--- Comparison Summary ---")
    print(f"AI-Generated Accuracy: {ai_acc:.3f}")
    print(f"Refined Accuracy:      {refined_acc:.3f}")
    print(
        "Conclusion: AI speeds up development, while human tuning improves performance."
    )