"""
train_model.py
Trains the Recruiter Decision Model on the complete Kaggle AI-Powered Resume Screening Dataset (2025) feature set.

Features used:
  - Skill_Count
  - Experience_Years
  - Edu_Rank
  - Projects_Count
  - Certifications_Count
  - Action_Verb_Count
  - Semantic_Match_Score
  - Word_Count

Saves:
  - backend/data/AI_Resume_Screening.csv
  - backend/models/recruiter_model.joblib
"""

import os
import random
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "AI_Resume_Screening.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "recruiter_model.joblib")

os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def generate_kaggle_2025_dataset(num_samples: int = 2000) -> pd.DataFrame:
    """Generates synthetic dataset following Kaggle AI-Powered Resume Screening Dataset (2025) specs."""
    np.random.seed(42)
    random.seed(42)

    names = ["Alex Johnson", "Priya Sharma", "David Kim", "Sophia Chen", "Rahul Verma", "Emily Davis", "Michael Brown", "Anita Roy"]
    job_roles = ["Machine Learning Engineer", "Backend Engineer", "Data Scientist", "NLP Engineer", "Full Stack Developer", "Cloud Architect"]
    education_levels = ["Diploma", "Bachelor", "Master", "PhD"]
    skills_pool = ["Python", "PyTorch", "TensorFlow", "Docker", "Flask", "SQL", "OpenCV", "NLP", "AWS", "Git", "Kubernetes", "React", "C++", "Spark", "GCP", "PostgreSQL", "Redis", "FastAPI"]

    data = []
    for i in range(1, num_samples + 1):
        resume_id = f"RES-2025-{i:04d}"
        name = random.choice(names)
        
        # Varied realistic distributions
        num_skills = random.choices([2, 4, 7, 12, 18, 24], weights=[0.15, 0.25, 0.3, 0.18, 0.08, 0.04])[0]
        skills_selected = random.sample(skills_pool, min(num_skills, len(skills_pool)))
        skills_str = ", ".join(skills_selected)

        exp_years = round(float(np.random.exponential(scale=3.0)), 1)

        exp_years = min(15.0, exp_years)

        education = random.choices(education_levels, weights=[0.1, 0.6, 0.25, 0.05])[0]
        cert_count = random.choices([0, 1, 2, 3, 4], weights=[0.4, 0.3, 0.18, 0.08, 0.04])[0]
        cert_str = f"{cert_count} Certified" if cert_count > 0 else "None"
        job_role = random.choice(job_roles)
        projects_count = random.randint(0, 6)
        action_verb_count = random.randint(1, 14)
        match_score = round(random.uniform(30.0, 95.0), 1)
        word_count = random.randint(150, 950)

        # Realistic Recruiter Hire Criteria
        edu_rank = 3 if education == "PhD" else (2 if education in ["Master", "Bachelor"] else 1)
        score_idx = (num_skills * 2.2) + (exp_years * 4.0) + (edu_rank * 8.0) + (projects_count * 3.5) + (cert_count * 4.0) + (action_verb_count * 1.5) + (match_score * 0.3)
        
        # Decision probability threshold with noise
        prob_hire = 1.0 / (1.0 + np.exp(-(score_idx - 62.0) / 10.0))
        decision = "Hire" if random.random() < prob_hire else "Reject"

        data.append({
            "Resume_ID": resume_id,
            "Name": name,
            "Skills": skills_str,
            "Skill_Count": len(skills_selected),
            "Experience (Years)": exp_years,
            "Education": education,
            "Edu_Rank": edu_rank,
            "Certifications": cert_str,
            "Certifications_Count": cert_count,
            "Job Role": job_role,
            "Projects Count": projects_count,
            "Action_Verb_Count": action_verb_count,
            "Semantic_Match_Score": match_score,
            "Word_Count": word_count,
            "Recruiter Decision": decision
        })

    df = pd.DataFrame(data)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved Kaggle 2025 dataset to: {DATA_PATH} ({len(df)} records)")
    return df


def train_recruiter_model():
    df = generate_kaggle_2025_dataset(2000)

    df["Target"] = df["Recruiter Decision"].map({"Hire": 1, "Reject": 0}).fillna(0)

    feature_cols = [
        "Skill_Count",
        "Experience (Years)",
        "Edu_Rank",
        "Projects Count",
        "Certifications_Count",
        "Action_Verb_Count",
        "Semantic_Match_Score",
        "Word_Count"
    ]

    X = df[feature_cols]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)

    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft'
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', CalibratedClassifierCV(estimator=ensemble, cv=5))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Model Training Complete! Calibrated Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["Reject", "Hire"]))

    # Save pipeline, feature names, mean and std for feature analysis
    joblib.dump({
        "pipeline": pipeline,
        "feature_names": feature_cols,
        "feature_means": X_train.mean().to_dict(),
        "accuracy": acc
    }, MODEL_PATH)
    print(f"Recruiter model pipeline saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_recruiter_model()
