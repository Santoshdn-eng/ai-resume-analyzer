# AI Resume Analyser and Job Match Platform

Upload a resume (PDF/DOCX/TXT) and get back:
- A parsed profile (name, email, phone, LinkedIn/GitHub, detected skills)
- A ranked list of job matches, scored by semantic similarity between the
  resume and each job description (via `sentence-transformers`)
- A skill-gap breakdown for each match (skills you have vs. skills the role wants)

## Project structure

```
ai-resume-analyzer/
├── backend/
│   ├── app.py               # Flask API + serves the frontend
│   ├── resume_parser.py      # PDF/DOCX/TXT -> structured fields
│   ├── matcher.py            # Embedding-based job matching engine
│   ├── skills_db.py          # Curated skills taxonomy
│   ├── data/sample_jobs.json # Sample job postings (swap for a real API later)
│   ├── templates/index.html  # Frontend (vanilla JS, no build step)
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
├── setup.sh                  # One-shot environment setup
└── PROJECT_PLAN.md           # Hour-by-hour build plan
```

## Quick start (local, no Docker)

```bash
bash setup.sh
source venv/bin/activate
cd backend
python app.py
```

Then open http://localhost:5000

> First run downloads the spaCy model (~13MB) and the sentence-transformer
> model (~80MB). After that, everything runs offline.

## Quick start (Docker)

```bash
docker compose up --build
```

Then open http://localhost:5000

## API reference

| Method | Endpoint        | Description                                  |
|--------|-----------------|-----------------------------------------------|
| GET    | `/api/health`   | Health check                                  |
| GET    | `/api/jobs`     | List the sample job dataset                   |
| POST   | `/api/analyze`  | Upload a resume (`multipart/form-data`, field `resume`) → parsed profile + ranked matches |

Example with `curl`:

```bash
curl -X POST http://localhost:5000/api/analyze \
  -F "resume=@/path/to/your_resume.pdf"
```

## Swapping in real job listings later

`backend/data/sample_jobs.json` is a static stand-in so the whole pipeline
works offline in a 10-hour build. To go further, replace `JobMatcher._load_jobs`
in `matcher.py` with a fetch from a live jobs API (e.g. Adzuna, Remotive, or a
scraped/aggregated source) and cache the embeddings.

## Opening in Antigravity

```bash
cd ai-resume-analyzer
antigravity .        # or: agy .   (Antigravity CLI)
```

If neither command is on your PATH, open Antigravity IDE directly and use
**File → Open Folder** to select the `ai-resume-analyzer` directory.
