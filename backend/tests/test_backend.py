"""
test_backend.py
Comprehensive test suite testing Flask endpoints, ML recruiter decision model, 
resume parser, matcher, and template engine.
"""

import unittest
import json
import os
import sys

# Add backend directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from recruiter_decision import RecruiterDecisionEngine
from matcher import JobMatcher
from template_engine import render_resume_template
from resume_parser import extract_skills, extract_contact_info


class TestBackendServices(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_health_endpoint(self):
        response = self.app.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data.get("status"), "ok")

    def test_jobs_endpoint(self):
        response = self.app.get("/api/jobs")
        self.assertEqual(response.status_code, 200)

    def test_recruiter_decision_engine(self):
        engine = RecruiterDecisionEngine()
        res = engine.predict(
            skill_count=12,
            match_score=85.0,
            exp_years=5.0,
            action_verbs=10,
            word_count=500,
            edu_rank=3,
            projects_count=4,
            certifications_count=2
        )
        self.assertIn("decision", res)
        self.assertIn("selected_percentage", res)
        self.assertGreaterEqual(res["selected_percentage"], 50.0)
        self.assertEqual(res["decision"], "Selected")

    def test_skill_extractor(self):
        text = "Experienced Machine Learning Engineer proficient in Python, PyTorch, Docker, SQL, and AWS."
        skills = extract_skills(text)
        self.assertIn("Python", skills)
        self.assertIn("PyTorch", skills)
        self.assertIn("Docker", skills)

    def test_contact_extractor(self):
        text = "Alex Johnson, Email: alex@example.com, Phone: +1-555-0199"
        contact = extract_contact_info(text)
        self.assertEqual(contact.get("email"), "alex@example.com")

    def test_template_rendering(self):
        resume_data = {
            "name": "Jane Doe",
            "title": "Software Architect",
            "skills": ["Python", "Flask", "Docker"]
        }
        html_out = render_resume_template("ats_classic", resume_data)
        self.assertIsInstance(html_out, str)
        self.assertIn("Jane Doe", html_out)


if __name__ == "__main__":
    unittest.main()
