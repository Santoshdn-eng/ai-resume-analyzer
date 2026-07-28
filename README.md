# 🚀 AI Resume Analyser & Job Match Platform

A full-stack, enterprise-grade AI application for resume parsing, automated ATS score optimization, machine learning recruiter decision prediction, real-time live job searching, and interactive AI resume consulting.

---

## 🌟 Key Features

1. **AI/ML Recruiter Selection Prediction (Kaggle 2025 Dataset)**:
   - Soft-voting ensemble machine learning model (`RandomForestClassifier` + `GradientBoostingClassifier` with `CalibratedClassifierCV`).
   - Evaluates **8 core screening features**: Skill Diversity, Experience Years, Education Rank, Projects Count, Certifications Count, Action Verbs, Semantic Match Score, and Word Count.
   - Computes **LIME Feature Impact Explanations** (positive & negative probability drivers).

2. **19+ ATS & Premier Academic Resume Templates**:
   - Includes **⭐ IIT Premier Academic Table Layout** (exact IIT Ropar / Bombay academic layout).
   - Includes **Nordic Minimal Glass**, **Cyberpunk Neo Tech**, **Oxford Royal Classic**, **Modern Split Sidebar**, **ATS Classic Standard (100% Taleo/Workday)**, and **ATS Harvard Corporate**.
   - **Direct MS Word-Style Screen Editing**: Click anywhere on the live resume preview to edit text, headings, or bullet points in real-time.

3. **Real-Time Live Job Search & Role Monitor**:
   - Queries live tech job API feeds with real-time timestamps (*"Posted 2 hours ago"*, *"Posted today"*).
   - Scores live jobs against candidate resumes using `sentence-transformers` (`all-MiniLM-L6-v2`) vector embeddings and cosine similarity.
   - Provides direct **Apply on LinkedIn ↗** links.

4. **OpenAI Resume Assistant & Chatbot**:
   - Interactive GPT-4o-mini chatbot consultant for bullet point rewriting, keyword suggestions, and selection score optimization.

---

## 🏗 Project Architecture

```
ai-resume-analyzer/
├── backend/
│   ├── app.py                # Flask REST API & Web UI server (Port 5050)
│   ├── resume_parser.py       # PDF/DOCX/TXT text parser, spaCy NER, and keyword extractor
│   ├── recruiter_decision.py  # Kaggle 2025 ML Ensemble classifier & LIME explainer
│   ├── job_search_engine.py   # Live job search, monitor & semantic scoring engine
│   ├── template_engine.py     # 19+ HTML resume template renderer & custom section builder
│   ├── matcher.py             # SentenceTransformer vector embedding & cosine matcher
│   ├── ai_integrator.py       # OpenAI GPT-4o-mini integration & interactive chatbot
│   ├── skills_db.py           # Curated tech skills taxonomy
│   ├── train_model.py         # Kaggle 2025 dataset generator & ensemble training script
│   ├── requirements.txt       # Production dependencies
│   ├── models/                # Trained ML model weights (`recruiter_model.joblib`)
│   └── templates/index.html   # Modern dark glassmorphism Web UI
├── Procfile                   # Cloud deployment config (Gunicorn)
├── render.yaml                # Render 1-click cloud deployment spec
├── docker-compose.yml         # Container orchestration spec
├── .gitignore                 # Version control exclusions
└── README.md                  # Project documentation
```

---

## ⚡ Quick Start

### 1. Run Locally (Python 3.12)

```bash
# Activate virtual environment
source venv/bin/activate

# Launch Flask App
cd backend
python app.py
```

Open your browser at **[http://localhost:5050](http://localhost:5050)**.

### 2. Run with Docker

```bash
docker compose up --build
```

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health check |
| `POST` | `/api/analyze` | Parse resume PDF/DOCX $\rightarrow$ Return recruiter ML decision, LIME breakdown & job matches |
| `POST` | `/api/jobs/live` | Fetch real-time open tech jobs & score match against candidate resume |
| `POST` | `/api/ai_suggest` | Generate OpenAI executive resume optimization report |
| `POST` | `/api/ai_chat` | Interactive AI Resume Consultant Chatbot |
| `POST` | `/api/template/render` | Render selected resume template HTML with custom sections |

---

## 👨‍💻 Author

**Santosh Debnath**  
AI/ML Engineer  
- **GitHub**: [github.com/Santoshdn-eng](https://github.com/Santoshdn-eng)  
- **LinkedIn**: [linkedin.com/in/Santosh-Debnath](https://linkedin.com/in/Santosh-Debnath)
