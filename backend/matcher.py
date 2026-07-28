"""
matcher.py
Semantic resume <-> job matching using sentence-transformers embeddings
and cosine similarity. Model is small (~80MB) and runs fine on CPU.
"""

import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"


class JobMatcher:
    def __init__(self, jobs_path: str):
        self.model = SentenceTransformer(MODEL_NAME)
        self.jobs = self._load_jobs(jobs_path)
        job_texts = [self._job_to_text(j) for j in self.jobs]
        self.job_embeddings = self.model.encode(job_texts, convert_to_numpy=True)

    @staticmethod
    def _load_jobs(jobs_path: str) -> list:
        with open(jobs_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _job_to_text(job: dict) -> str:
        skills = ", ".join(job.get("required_skills", []))
        return f"{job['title']}. {job['description']} Required skills: {skills}"

    def match(self, resume_text: str, resume_skills: list, top_n: int = 5) -> list:
        resume_embedding = self.model.encode([resume_text], convert_to_numpy=True)
        scores = cosine_similarity(resume_embedding, self.job_embeddings)[0]

        ranked_idx = scores.argsort()[::-1][:top_n]
        results = []
        resume_skills_set = {s.lower() for s in resume_skills}

        for idx in ranked_idx:
            job = self.jobs[idx]
            required = job.get("required_skills", [])
            required_set = {s.lower() for s in required}

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
