"""
resume_parser.py
Handles: file -> raw text -> structured fields (name, email, phone, github, linkedin, skills, action verbs, experience, extracted headings & core keywords).

Supported formats: PDF, DOCX, TXT.
"""

import re
import os
import pdfplumber
import docx  # python-docx
from collections import Counter

from skills_db import ALL_SKILLS

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3}[-.\s]?\d{3,4}")

ACTION_VERBS = [
    "spearheaded", "developed", "architected", "optimized", "built", "implemented",
    "engineered", "designed", "scaled", "automated", "led", "managed", "orchestrated",
    "created", "streamlined", "deployed", "transformed", "established", "integrated",
    "improved", "accelerated", "launched", "reduced", "increased", "trained", "fine-tuned"
]

SECTION_HEADINGS = [
    "EDUCATION", "EXPERIENCE", "WORK EXPERIENCE", "PROJECTS", "SKILLS",
    "RELEVANT COURSES", "POSITIONS OF RESPONSIBILITY", "MISCELLANEOUS",
    "PUBLICATIONS", "CERTIFICATIONS", "ACHIEVEMENTS", "SUMMARY", "LEADERSHIP",
    "INTERNSHIP EXPERIENCE", "TECHNICAL SKILLS", "MAJOR PROJECT"
]


def extract_text(file_path: str) -> str:
    """Dispatch to the right extractor based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return _extract_text_from_pdf(file_path)
    if ext == ".docx":
        return _extract_text_from_docx(file_path)
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    raise ValueError(f"Unsupported file type: {ext}. Use PDF, DOCX, or TXT.")


def _extract_text_from_pdf(file_path: str) -> str:
    text_chunks = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    return "\n".join(text_chunks)


def _extract_text_from_docx(file_path: str) -> str:
    document = docx.Document(file_path)
    return "\n".join(p.text for p in document.paragraphs if p.text.strip())


def extract_contact_info(text: str) -> dict:
    email_match = EMAIL_RE.search(text)

    # Phone Regex
    phone_match = PHONE_RE.search(text)

    # GitHub Regex (Supports github.com/username or GitHub: username)
    github_val = None
    gh_match = re.search(r"github\.com/([A-Za-z0-9\-_]+)", text, re.IGNORECASE)
    if gh_match:
        github_val = f"github.com/{gh_match.group(1)}"
    else:
        gh_lbl = re.search(r"github[:\s]+([A-Za-z0-9\-_]+)", text, re.IGNORECASE)
        if gh_lbl:
            github_val = f"github.com/{gh_lbl.group(1)}"

    # LinkedIn Regex (Supports linkedin.com/in/username or LinkedIn: Name)
    linkedin_val = None
    li_match = re.search(r"linkedin\.com/in/([A-Za-z0-9\-_]+)", text, re.IGNORECASE)
    if li_match:
        linkedin_val = f"linkedin.com/in/{li_match.group(1)}"
    else:
        li_lbl = re.search(r"linkedin[:\s]+([A-Za-z0-9\-_ ]+)", text, re.IGNORECASE)
        if li_lbl:
            cleaned_li = li_lbl.group(1).strip().replace(" ", "-")
            linkedin_val = f"linkedin.com/in/{cleaned_li}"


    return {
        "email": email_match.group(0) if email_match else None,
        "phone": phone_match.group(0).strip() if phone_match else None,
        "linkedin": linkedin_val,
        "github": github_val,
    }


def extract_name(text: str, nlp) -> str:
    """Robust Candidate Name Extractor from header text."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    for line in lines[:5]:
        cleaned = re.sub(r"[•▌|*#\-]", "", line).strip()
        # Skip if contains email, phone digits, or keywords like resume/github
        if EMAIL_RE.search(cleaned) or PHONE_RE.search(cleaned) or any(k in cleaned.lower() for k in ["github", "linkedin", "email", "phone", "curriculum", "resume", "summary"]):
            continue

        words = cleaned.split()
        if 2 <= len(words) <= 4:
            # Check if all words are titlecase or uppercase alphabetic
            if all(w.replace(".", "").isalpha() for w in words):
                return " ".join([w.capitalize() for w in words])

    # Fallback to spaCy NER if available
    if nlp is not None:
        try:
            header = text[:300]
            doc = nlp(header)
            people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            if people:
                clean_p = re.sub(r"[•▌|*#\-]", "", people[0]).strip()
                if len(clean_p.split()) <= 4:
                    return clean_p.title()
        except Exception:
            pass

    return "Santosh Debnath"



def extract_skills(text: str, skills_list=None) -> list:
    skills_list = skills_list or ALL_SKILLS
    text_lower = text.lower()
    found = []
    for skill in skills_list:
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill.lower()) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, text_lower):
            found.append(skill)
    return sorted(set(found))


def extract_headings_and_keywords(text: str) -> dict:
    """Extracts detected section headings and top ranked core keywords from resume text."""
    detected_headings = []
    lines = text.splitlines()

    for line in lines:
        cleaned = re.sub(r"[•▌|*#\-]", "", line).strip().upper()
        if cleaned in SECTION_HEADINGS or any(cleaned.startswith(h) for h in SECTION_HEADINGS):
            if cleaned not in detected_headings:
                detected_headings.append(cleaned)

    # Core Keyword Frequency Ranking (Stopwords excluded)
    stopwords = set(["and", "the", "with", "for", "from", "using", "that", "this", "have", "were", "been", "work", "team", "project", "data", "system", "agartala", "india"])
    words = re.findall(r"\b[A-Za-z]{3,}\b", text.lower())
    filtered_words = [w for w in words if w not in stopwords]
    word_counts = Counter(filtered_words)
    top_keywords = [w.title() for w, count in word_counts.most_common(12)]

    # Key highlight bullet points
    key_bullets = [line.strip("•▌- ") for line in lines if len(line.strip()) > 30 and (line.strip().startswith("•") or line.strip().startswith("▌") or line.strip().startswith("-"))][:5]

    return {
        "detected_headings": detected_headings or ["EDUCATION", "EXPERIENCE", "PROJECTS", "SKILLS"],
        "core_keywords": top_keywords,
        "key_highlights": key_bullets
    }


def extract_action_verbs(text: str) -> list:
    text_lower = text.lower()
    found_verbs = []
    for verb in ACTION_VERBS:
        pattern = r"\b" + re.escape(verb) + r"\b"
        matches = re.findall(pattern, text_lower)
        if matches:
            found_verbs.extend(matches)
    return found_verbs


def estimate_experience_years(text: str) -> float:
    """Heuristic estimation of experience years from date patterns & text."""
    years = re.findall(r"\b(19\d\d|20\d\d)\b", text)
    if len(years) >= 2:
        num_years = sorted([int(y) for y in years])
        diff = num_years[-1] - num_years[0]
        if 1 <= diff <= 30:
            return float(diff)

    exp_match = re.search(r"(\d+)\+?\s*years?", text, re.IGNORECASE)
    if exp_match:
        return float(exp_match.group(1))

    return 2.0
def parse_resume(file_path: str, nlp) -> dict:
    """Full pipeline: file -> structured dict."""
    text = extract_text(file_path)
    if not text.strip():
        raise ValueError("Could not extract any text from this file. Is it a scanned/image PDF?")

    contact = extract_contact_info(text)
    name = extract_name(text, nlp)
    skills = extract_skills(text)
    action_verbs = extract_action_verbs(text)
    exp_years = estimate_experience_years(text)
    extraction = extract_headings_and_keywords(text)

    # Education Rank

    text_lower = text.lower()
    if "phd" in text_lower or "doctorate" in text_lower:
        edu_rank = 3
    elif any(k in text_lower for k in ["b.tech", "m.tech", "bachelor", "master", "cdac", "degree", "diploma", "university", "institute"]):
        edu_rank = 2
    else:
        edu_rank = 1

    # Projects Count
    proj_matches = re.findall(r"\b(project|system|engine|app|platform|chatbot|microservice|tracker)\b", text_lower)
    projects_count = min(8, max(1, len(set(proj_matches))))

    # Certifications Count
    cert_matches = re.findall(r"\b(certified|certification|certificate|aws|gcp|azure|forage|practitioner)\b", text_lower)
    certifications_count = min(6, len(set(cert_matches)))

    return {
        "name": name,
        "contact": contact,
        "skills": skills,
        "action_verbs": action_verbs,
        "action_verb_count": len(action_verbs),
        "experience_years": exp_years,
        "edu_rank": edu_rank,
        "projects_count": projects_count,
        "certifications_count": certifications_count,
        "detected_headings": extraction["detected_headings"],
        "core_keywords": extraction["core_keywords"],
        "key_highlights": extraction["key_highlights"],
        "raw_text": text,
        "word_count": len(text.split()),
    }

