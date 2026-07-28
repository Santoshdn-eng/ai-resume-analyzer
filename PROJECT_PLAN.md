# AI Resume Analyser and Job Match Platform — 10-Hour Build Plan

## 1. Scope decision (read this first)

A full production job-matching platform is a multi-week project. To actually
ship something working in 10 hours, the plan below deliberately **cuts**
three things that would otherwise eat your whole day:

| Cut | Why | What you do instead |
|---|---|---|
| Training a custom ML model | Training + evaluation alone takes hours | Use pre-trained `sentence-transformers` (all-MiniLM-L6-v2) for matching, and spaCy's built-in NER for names — zero training required |
| Live job board integration | OAuth/API keys/rate limits are a time sink | Ship with a static `sample_jobs.json` (12 realistic postings, included). Swapping in a real API is a documented stretch goal, not a blocker |
| Full React build | CRA/Vite setup + component wiring adds 1-2 hours of scaffolding | Single-file HTML/CSS/vanilla JS frontend, no build step, served directly by Flask |

This isn't cutting corners on quality — it's the difference between a demo
that works end-to-end at hour 10 and one that's still half-wired.

## 2. Architecture

```
Browser (drag & drop resume)
        │  POST /api/analyze (multipart file)
        ▼
Flask app (app.py)
        │
        ├─ resume_parser.py  → extract text (pdfplumber / python-docx)
        │                      → contact info (regex)
        │                      → name (spaCy NER)
        │                      → skills (keyword match vs skills_db.py)
        │
        └─ matcher.py        → embed resume text (sentence-transformers)
                              → cosine similarity vs. pre-embedded jobs
                              → rank top N + skill-gap diff
        ▼
JSON response → rendered in the browser (profile card + ranked matches)
```

Everything runs in one Flask process. No database, no message queue, no
auth — none of that is needed for a working demo, and adding it now is how
projects blow their time budget.

## 3. Tech stack

- **Backend:** Flask + Flask-Cors (matches your existing capstone stack)
- **Resume parsing:** `pdfplumber` (PDF), `python-docx` (DOCX), stdlib `re` for email/phone
- **NLP:** spaCy `en_core_web_sm` for name extraction, keyword matching for skills
- **Matching:** `sentence-transformers` (`all-MiniLM-L6-v2`) + `scikit-learn` cosine similarity
- **Frontend:** plain HTML/CSS/JS (no framework, no build step)
- **Deployment:** Docker + docker-compose

## 4. Hour-by-hour plan

**Hour 1 — Environment + scaffolding**
- Run `setup.sh` (venv, pip installs, spaCy model download, git init)
- Open the project in Antigravity, confirm `python app.py` boots and
  `/api/health` returns `{"status": "ok"}`
- Commit: "project scaffold running"

**Hour 2 — Resume text extraction**
- Wire up `resume_parser.extract_text` for PDF/DOCX/TXT
- Test against 2-3 real resumes (yours + a couple of sample ones) — PDF
  text extraction quality varies a lot by how the PDF was generated, so
  check this early, not at hour 9
- Commit: "resume text extraction working"

**Hour 3 — Contact info + skills extraction**
- Regex for email/phone/LinkedIn/GitHub (already scaffolded)
- Tune `skills_db.py` — add any skills specific to the resumes you're
  testing with that aren't already in the list
- Commit: "contact + skill extraction"

**Hour 4-5 — Matching engine**
- Confirm `sentence-transformers` downloads and embeds correctly
- Sanity-check scores: feed in a resume you know is a strong fit for
  "Machine Learning Engineer" and a weak fit for "Cloud & DevOps Engineer" —
  confirm the ranking reflects that
- Add the skill-gap diff (matched vs. missing) — already scaffolded in
  `matcher.py`, verify output format
- Commit: "job matching engine working end-to-end"

**Hour 6 — Wire the API**
- `/api/analyze` end-to-end: upload → parse → match → JSON response
- Handle the obvious failure modes: no file, wrong file type, scanned/image
  PDF with no extractable text, oversized file
- Test with `curl` before touching the frontend — isolates backend bugs
  from frontend bugs
- Commit: "API endpoint complete with error handling"

**Hour 7 — Frontend**
- Drag-and-drop upload, loading state, error state (all scaffolded in
  `templates/index.html`)
- Wire the results render: profile card, top-match gauge, other matches list
- Commit: "frontend wired to API"

**Hour 8 — Polish the analysis quality**
- This is where a demo goes from "technically works" to "actually
  impressive": review 3-4 test resumes end-to-end and fix anything that
  looks obviously wrong (a garbled name, a missed obvious skill, a nonsense
  top match)
- Add a resume "completeness" note if word count is very low (parsing
  likely failed) — one line in `app.py`'s response is enough
- Commit: "analysis quality pass"

**Hour 9 — Dockerize + test the container path**
- `docker compose up --build`, confirm it serves on :5000 identically to
  the local run
- This matters because "works on my machine" is the #1 failure mode when
  demoing later on a different laptop or projector
- Commit: "dockerized"

**Hour 10 — README, demo prep, buffer**
- Finalize README with exact run steps (already scaffolded)
- Prepare 2-3 sample resumes you'll use live in the demo — do **not** rely
  on finding one on the spot during a live demo
- Push to GitHub
- Keep the last 20-30 minutes as unstructured buffer — something in a
  10-hour build always takes longer than planned; this is normal, not a
  sign you're behind

## 5. Terminal setup (recap)

```bash
cd ai-resume-analyzer
bash setup.sh
source venv/bin/activate
cd backend && python app.py
```

## 6. Stretch goals (only if you finish early)

In priority order — each is genuinely separable, so stop wherever your
clock runs out:

1. Resume "match score" summary line using an LLM (e.g. via Hugging Face
   Inference API) to generate a 2-sentence fit explanation for the top match
2. Swap `sample_jobs.json` for a real jobs API (Adzuna, Remotive)
3. Export the analysis as a downloadable PDF report
4. Multi-resume comparison (batch upload, compare candidates against one job)
5. Auth + persistence (only worth it if this becomes a real ongoing project)

## 7. Common time sinks to avoid

- **Don't** try to fine-tune or train anything — pre-trained embeddings are
  the entire point of hitting this in 10 hours
- **Don't** start the React rewrite "just to make it look nicer" — the
  provided frontend already covers drag-and-drop, loading states, and
  results rendering
- **Don't** debug PDF parsing edge cases exhaustively — pdfplumber handles
  the vast majority of standard resumes; if one specific PDF fails, test
  with a different sample rather than burning an hour on one file
- **Don't** add a database — the app is stateless by design; resumes are
  deleted right after analysis (see the `finally` block in `app.py`)
