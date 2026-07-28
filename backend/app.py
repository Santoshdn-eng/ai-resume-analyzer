"""
app.py
Flask backend for the AI Resume Analyser and Job Match Platform.

Endpoints:
  GET  /                     -> serves the frontend
  GET  /api/health            -> health check
  GET  /api/jobs              -> lists job catalog dataset
  POST /api/analyze           -> parse resume + rank jobs + recruiter decision (Selected/Rejected %) + LIME feature breakdown
  POST /api/ai_suggest        -> OpenAI / Anthropic / Local AI feedback + missing keywords + bullet rewrites
  POST /api/template/render   -> renders auto-edited HTML resume template
  GET  /api/eda_insights      -> returns Kaggle dataset EDA metrics
"""

import os
import uuid
import spacy
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

from resume_parser import parse_resume
from matcher import JobMatcher
from recruiter_decision import RecruiterDecisionEngine
from ai_integrator import get_ai_suggestions
from template_engine import render_resume_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
JOBS_PATH = os.path.join(BASE_DIR, "data", "sample_jobs.json")
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Load ML Models once at startup
print("Loading spaCy model...")
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"spaCy model load fallback: {e}")
    nlp = None


print("Loading sentence-transformers & job dataset...")
matcher = JobMatcher(JOBS_PATH)

print("Initializing Recruiter Decision Classifier & LIME engine...")
recruiter_engine = RecruiterDecisionEngine()
print("All AI & ML Services Ready.")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/jobs")
def list_jobs():
    return jsonify(matcher.jobs)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded. Attach a file under the 'resume' field."}), 400

    file = request.files["resume"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Upload a PDF, DOCX, or TXT file."}), 400

    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    try:
        parsed = parse_resume(filepath, nlp)
        top_n = int(request.form.get("top_n", 6))
        matches = matcher.match(parsed["raw_text"], parsed["skills"], top_n=top_n)

        top_match_score = matches[0]["score"] if matches else 50.0

        # Run Recruiter Decision Classifier + LIME feature explainer (8 Kaggle features)
        decision_results = recruiter_engine.predict(
            skill_count=len(parsed["skills"]),
            match_score=top_match_score,
            exp_years=parsed.get("experience_years", 2.0),
            action_verbs=parsed.get("action_verb_count", 3),
            word_count=parsed["word_count"],
            edu_rank=parsed.get("edu_rank", 2),
            projects_count=parsed.get("projects_count", 3),
            certifications_count=parsed.get("certifications_count", 1)
        )


        response = {
            "name": parsed["name"],
            "contact": parsed["contact"],
            "skills": parsed["skills"],
            "action_verbs": parsed.get("action_verbs", []),
            "word_count": parsed["word_count"],
            "experience_years": parsed.get("experience_years", 2.5),
            "matches": matches,
            "recruiter_decision": decision_results,
        }
        return jsonify(response)

    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        app.logger.exception("Unexpected error during analysis")
        return jsonify({"error": "Something went wrong while analyzing the resume.", "detail": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


from ai_integrator import get_ai_suggestions, chat_with_ai


@app.route("/api/ai_suggest", methods=["POST"])
def ai_suggest():
    data = request.json or {}
    candidate_data = data.get("candidate", {})
    target_job = data.get("target_job", {})
    provider = data.get("provider", "auto")
    api_key = data.get("api_key", None)

    try:
        suggestions = get_ai_suggestions(candidate_data, target_job, provider=provider, api_key=api_key)
        return jsonify(suggestions)
    except Exception as e:
        app.logger.exception("AI Suggestion generation error")
        return jsonify({"error": "Failed to generate AI suggestions.", "detail": str(e)}), 500


@app.route("/api/ai_chat", methods=["POST"])
def ai_chat():
    data = request.json or {}
    user_message = data.get("message", "")
    chat_history = data.get("history", [])
    candidate_data = data.get("candidate", {})
    target_job = data.get("target_job", {})
    api_key = data.get("api_key", None)

    if not user_message:
        return jsonify({"error": "Message is required."}), 400

    try:
        response = chat_with_ai(
            user_message=user_message,
            chat_history=chat_history,
            candidate_data=candidate_data,
            target_job=target_job,
            api_key=api_key
        )
        return jsonify(response)
    except Exception as e:
        app.logger.exception("AI Chat error")
        return jsonify({"error": "Failed to generate response.", "detail": str(e)}), 500


from job_search_engine import search_and_match_live_jobs



@app.route("/api/jobs/live", methods=["POST", "GET"])
def live_jobs():
    data = request.json if request.is_json else request.args
    query = data.get("query", "Machine Learning")
    resume_text = data.get("raw_text", None)
    resume_skills = data.get("skills", [])

    try:
        matched_live_jobs = search_and_match_live_jobs(
            query=query,
            matcher_instance=matcher,
            resume_text=resume_text,
            resume_skills=resume_skills,
            limit=8
        )
        return jsonify({"jobs": matched_live_jobs, "query": query, "timestamp": "Live real-time feed"})
    except Exception as e:
        app.logger.exception("Live job search error")
        return jsonify({"error": "Failed to fetch live jobs.", "detail": str(e)}), 500




@app.route("/api/template/render", methods=["POST"])
def render_template_route():
    data = request.json or {}
    template_id = data.get("template_id", "modern_tech")
    candidate_data = data.get("candidate_data", {})

    try:
        rendered_html = render_resume_template(template_id, candidate_data)
        return jsonify({"html": rendered_html})
    except Exception as e:
        return jsonify({"error": "Failed to render resume template.", "detail": str(e)}), 500


@app.route("/api/eda_insights")
def eda_insights():
    return jsonify({
        "dataset_name": "AI Power Resume Screening Dataset 2025 (Kaggle)",
        "total_samples": 2500,
        "selection_rate": "38.4% Selected vs 61.6% Rejected",
        "key_correlations": [
            {"feature": "Skill Match %", "correlation": 0.72},
            {"feature": "Relevant Experience Years", "correlation": 0.64},
            {"feature": "Action Verbs Count", "correlation": 0.48},
            {"feature": "Resume Word Count (300-800 words)", "correlation": 0.35}
        ]
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5050))
    print(f"Starting server on http://localhost:{port} ...")
    app.run(debug=True, host="0.0.0.0", port=port)

