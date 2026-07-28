"""
skills_db.py
A lightweight, hand-curated skills taxonomy used for:
  1. Extracting skills mentioned in a resume (keyword matching)
  2. Comparing resume skills against a job's required skills (gap analysis)

This intentionally avoids training a custom NER model — with a 10 hour
budget, a curated keyword list + spaCy's built-in NER for names/orgs gets
you 90% of the value for a fraction of the effort.

Add/remove skills freely; matching is case-insensitive and word-boundary
aware (see resume_parser.extract_skills).
"""

SKILLS_DB = {
    "languages": [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C", "C#", "Go",
        "Rust", "R", "SQL", "PHP", "Kotlin", "Swift", "Scala", "MATLAB", "Bash",
    ],
    "ml_ai": [
        "Machine Learning", "Deep Learning", "PyTorch", "TensorFlow", "Keras",
        "Scikit-learn", "OpenCV", "YOLO", "NLP", "Computer Vision",
        "Hugging Face", "Transformers", "LangChain", "LLM", "Sentence Transformers",
        "Reinforcement Learning", "Pandas", "NumPy", "spaCy",
    ],
    "web_backend": [
        "Flask", "Django", "FastAPI", "REST API", "GraphQL", "Node.js",
        "Express.js", "Spring Boot", "React", "React.js", "Vue.js", "Angular",
        "HTML", "CSS", "Tailwind CSS", "Redux",
    ],
    "data": [
        "PySpark", "Apache Spark", "Hadoop", "ETL", "Data Analysis",
        "Data Visualization", "Tableau", "Power BI", "Airflow",
    ],
    "databases": [
        "MySQL", "PostgreSQL", "MongoDB", "Redis", "SQLite", "Firebase",
        "Elasticsearch", "DynamoDB",
    ],
    "cloud_devops": [
        "AWS", "GCP", "Azure", "Docker", "Kubernetes", "CI/CD", "Jenkins",
        "GitHub Actions", "Terraform", "Linux", "Nginx", "Git", "GitHub",
    ],
    "other": [
        "Agile", "Scrum", "Project Management", "Data Structures",
        "Algorithms", "System Design", "Testing", "Unit Testing",
    ],
}

# Flat list for quick iteration
ALL_SKILLS = sorted({s for group in SKILLS_DB.values() for s in group})
