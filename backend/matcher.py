"""
matcher.py
Semantic resume <-> job matching using sentence-transformers embeddings
and cosine similarity. Optimized for instant startup on low-resource cloud environments (Render / Railway).
"""

import json
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"


class JobMatcher:
    def __init__(self, jobs_path: str):
        self.jobs_path = jobs_path
        self.jobs = self._load_jobs(jobs_path)
        self._model = None
        self._job_embeddings = None

    @staticmethod
    def _load_jobs(jobs_path: str) -> list:
        with open(jobs_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _job_to_text(job: dict) -> str:
        skills = ", ".join(job.get("required_skills", []))
        return f"{job['title']}. {job['description']} Required skills: {skills}"

    def _ensure_loaded(self):
        if self._model is None:
            print("Lazy-loading SentenceTransformer model...")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(MODEL_NAME)
            job_texts = [self._job_to_text(j) for j in self.jobs]
            self._job_embeddings = self._model.encode(job_texts, convert_to_numpy=True)

    def match(self, resume_text: str, resume_skills: list, top_n: int = 5) -> list:
        self._ensure_loaded()
        resume_embedding = self._model.encode([resume_text], convert_to_numpy=True)
        scores = cosine_similarity(resume_embedding, self._job_embeddings)[0]

        ranked_idx = scores.argsort()[::-1][:top_n]
        results = []
        resume_skills_set = {s.lower() for s in resume_skills}

        for idx in ranked_idx:
            job = self.jobs[idx]
            required = job.get("required_skills", [])

            matched = [s for s in required if s.lower() in resume_skills_set]
            missing = [s for s in required if s.lower() not in resume_skills_set]

            results.append({
                "job_id": job["id"],
                "title": job["title"],
                "company": job["company"],
                "location": job.get("location", "Remote"),
                "score": round(float(scores[idx]) * 100, 1),
                "matched_skills": matched,
                "missing_skills": missing,
                "skill_coverage": round(len(matched) / len(required) * 100, 1) if required else 0,
            })
        return results
