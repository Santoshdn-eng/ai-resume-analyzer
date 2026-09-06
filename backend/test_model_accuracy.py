"""
test_model_accuracy.py
Script to test and verify the Recruiter Decision ML Model accuracy, 
evaluate performance metrics (Accuracy, ROC-AUC, Confusion Matrix), 
and test inference on candidate profiles.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from recruiter_decision import RecruiterDecisionEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "AI_Resume_Screening.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "recruiter_model.joblib")


def evaluate_saved_model():
    print("=" * 60)
    print("1. EVALUATING SAVED MODEL ON KAGGLE DATASET")
    print("=" * 60)

    if not os.path.exists(DATA_PATH) or not os.path.exists(MODEL_PATH):
        print("Model or dataset not found. Run train_model.py first.")
        return

    # Load dataset & model pipeline
    df = pd.read_csv(DATA_PATH)
    saved = joblib.load(MODEL_PATH)
    pipeline = saved["pipeline"]
    feature_cols = saved["feature_names"]

    X = df[feature_cols]
    y = df["Recruiter Decision"].map({"Hire": 1, "Reject": 0}).fillna(0)

    # Predict
    y_pred = pipeline.predict(X)
    y_probs = pipeline.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_probs)
    cm = confusion_matrix(y, y_pred)

    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print(f"ROC-AUC Score:    {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"  True Negatives (Correct Reject): {cm[0][0]}")
    print(f"  False Positives (Wrong Hire):    {cm[0][1]}")
    print(f"  False Negatives (Wrong Reject):  {cm[1][0]}")
    print(f"  True Positives (Correct Hire):   {cm[1][1]}")

    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Reject", "Hire"]))


def test_candidate_profiles():
    print("\n" + "=" * 60)
    print("2. TESTING INFERENCE ON SYNTHETIC CANDIDATE PROFILES")
    print("=" * 60)

    engine = RecruiterDecisionEngine()

    test_cases = [
        {
            "name": "Senior ML Engineer (Strong Profile)",
            "params": {
                "skill_count": 15,
                "match_score": 88.5,
                "exp_years": 6.0,
                "action_verbs": 12,
                "word_count": 550,
                "edu_rank": 3,  # PhD
                "projects_count": 5,
                "certifications_count": 3
            }
        },
        {
            "name": "Mid-Level Backend Dev (Moderate Profile)",
            "params": {
                "skill_count": 7,
                "match_score": 62.0,
                "exp_years": 2.5,
                "action_verbs": 5,
                "word_count": 400,
                "edu_rank": 2,  # Bachelor/Master
                "projects_count": 2,
                "certifications_count": 1
            }
        },
        {
            "name": "Junior Unqualified Applicant (Weak Profile)",
            "params": {
                "skill_count": 2,
                "match_score": 32.0,
                "exp_years": 0.2,
                "action_verbs": 1,
                "word_count": 180,
                "edu_rank": 1,  # Diploma
                "projects_count": 0,
                "certifications_count": 0
            }
        }
    ]

    for tc in test_cases:
        res = engine.predict(**tc["params"])
        print(f"\n📌 Candidate: {tc['name']}")
        print(f"   Decision:            {res['decision']}")
        print(f"   Selection Probability: {res['selected_percentage']}%")
        print(f"   Rejection Probability: {res['rejected_percentage']}%")
        print("   Key Drivers (LIME Feature Impact):")
        for exp in res["explanations"]:
            symbol = "🟢 +" if exp["impact"] == "positive" else "🔴 "
            print(f"     {symbol} {exp['feature']} ({exp['weight']}%): {exp['description']}")


if __name__ == "__main__":
    evaluate_saved_model()
    test_candidate_profiles()
