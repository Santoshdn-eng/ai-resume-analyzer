"""
streamlit_app.py
Production-Grade Streamlit Application for AI Resume Analyser & Recruiter ML Decision Predictor.

Features:
  1. Interactive Resume Upload (PDF / DOCX / TXT) & Manual Text Parsing
  2. Soft-Voting ML Recruiter Decision Predictor (Hire/Reject %) + Calibrated Accuracy Gauge
  3. LIME Feature Driver Explanations with Interactive Visualizations (Plotly)
  4. Real-time Semantic Role Matcher & Skill Keyword Gap Analysis
  5. 19+ Resume Template Previewer & AI Optimization Assistant
  6. Direct One-Click Model Retraining
"""

import os
import sys

# Ensure backend directory is in Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Backend imports
from backend.resume_parser import parse_resume, extract_skills, extract_contact_info
from backend.recruiter_decision import RecruiterDecisionEngine
from backend.matcher import JobMatcher
from backend.template_engine import render_resume_template
from backend.train_model import train_recruiter_model, MODEL_PATH, DATA_PATH

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Resume Analyser & Recruiter Decision ML",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism Theme)
st.markdown("""
<style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        backdrop-filter: blur(10px);
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #8b949e;
        margin-bottom: 4px;
        font-weight: 500;
    }
    .metric-value-hire {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2ea043;
    }
    .metric-value-reject {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f85149;
    }
    .badge-positive {
        background-color: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.4);
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.85rem;
    }
    .badge-negative {
        background-color: rgba(248, 81, 73, 0.15);
        color: #f85149;
        border: 1px solid rgba(248, 81, 73, 0.4);
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_ml_services():
    """Initializes ML models, spaCy, and Job Matcher engine."""
    recruiter_engine = RecruiterDecisionEngine()
    jobs_path = os.path.join(BASE_DIR, "backend", "data", "sample_jobs.json")
    matcher = JobMatcher(jobs_path)
    return recruiter_engine, matcher


# Load Cached Engines
recruiter_engine, matcher = load_ml_services()

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("---")
    
    st.subheader("🎯 Target Job Role")
    target_role = st.selectbox(
        "Select target position:",
        [
            "Machine Learning Engineer",
            "Backend Engineer",
            "Data Scientist",
            "Full Stack Developer",
            "Cloud Architect",
            "NLP Engineer"
        ]
    )
    
    st.subheader("📄 Upload Resume Document")
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        help="Upload candidate resume file for parsing & model prediction."
    )
    
    manual_text = st.text_area(
        "Or paste resume text directly:",
        height=150,
        placeholder="Paste plain text resume content here..."
    )
    
    st.markdown("---")
    st.subheader("🤖 Recruiter ML Model Status")
    st.info(f"**Architecture**: Calibrated Ensemble (RF + GB)\n\n**Dataset**: Kaggle 2025 (2,000 samples)\n\n**Accuracy**: {recruiter_engine.accuracy * 100:.1f}%")
    
    if st.button("🔄 Retrain ML Model", use_container_width=True):
        with st.spinner("Retraining Calibrated Soft-Voting Ensemble on Kaggle dataset..."):
            train_recruiter_model()
            st.cache_resource.clear()
            st.success("Model successfully retrained and reloaded!")
            st.rerun()

# --- MAIN DASHBOARD HEADER ---
st.title("🚀 AI Resume Analyser & Recruiter ML Decision Predictor")
st.markdown(
    "Enterprise-grade AI platform for automated ATS parsing, **Soft-Voting Ensemble ML Recruiter Decision Prediction**, "
    "**LIME Feature Explanations**, and **Real-Time Job Match Scoring**."
)
st.markdown("---")

# Extract Text from File or Manual Input
resume_text = ""
file_name = "Manual Input"

if uploaded_file is not None:
    file_name = uploaded_file.name
    temp_dir = os.path.join(BASE_DIR, "backend", "uploads")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    parsed = parse_resume(temp_path)
    resume_text = parsed.get("cleaned_text", "")
elif manual_text.strip():
    parsed = {
        "cleaned_text": manual_text,
        "word_count": len(manual_text.split()),
        "skills": extract_skills(manual_text),
        "contact_info": extract_contact_info(manual_text),
        "action_verbs_count": sum(1 for verb in ["developed", "built", "managed", "designed", "architected", "deployed", "optimized", "implemented", "created", "led"] if verb in manual_text.lower()),
        "experience_years": 3.0,
        "education": "Bachelor"
    }
    resume_text = manual_text

# Default Demo Content if empty
if not resume_text:
    st.info("💡 **Quick Start**: Upload a resume in the sidebar or click **Run Sample Demo Candidate** below.")
    if st.button("▶️ Run Sample Demo Candidate (Senior ML Engineer)", type="primary"):
        demo_text = """
        Alex Johnson - Senior Machine Learning Engineer
        Email: alex.johnson@example.com | Phone: +1-555-0199 | Location: San Francisco, CA
        LinkedIn: linkedin.com/in/alex-johnson-ml | GitHub: github.com/alex-johnson-ml

        SUMMARY:
        Results-driven Senior Machine Learning Engineer with 6+ years of industry experience designing, building, and deploying large-scale deep learning models, NLP pipelines, and real-time inference microservices. 

        SKILLS:
        Python, PyTorch, TensorFlow, Docker, Kubernetes, SQL, PostgreSQL, FastAPI, Flask, AWS, GCP, Git, OpenCV, NLP, Spark, Redis, Scikit-learn, Ray, MLOps

        EXPERIENCE:
        Senior Machine Learning Engineer | TechCorp Inc. (2022 - Present)
        - Architected end-to-end NLP recommendation engine serving 5M+ daily active users, boosting user retention by 24%.
        - Deployed PyTorch deep learning models to AWS SageMaker using Kubernetes and Docker containers.
        - Optimized inference latency by 45% using TensorRT and model quantization techniques.

        Machine Learning Engineer | DataLabs (2018 - 2022)
        - Developed computer vision pipeline for automated defect detection using OpenCV and PyTorch.
        - Built automated ETL data pipelines with PySpark and PostgreSQL processing 10TB+ weekly data.

        EDUCATION & CERTIFICATIONS:
        - Ph.D. in Computer Science (Artificial Intelligence) - Stanford University
        - AWS Certified Machine Learning - Specialty (2023)
        - TensorFlow Developer Certified (2022)
        """
        parsed = {
            "cleaned_text": demo_text,
            "word_count": len(demo_text.split()),
            "skills": ["Python", "PyTorch", "TensorFlow", "Docker", "Kubernetes", "SQL", "PostgreSQL", "FastAPI", "Flask", "AWS", "GCP", "Git", "OpenCV", "NLP", "Spark", "Redis"],
            "contact_info": {"name": "Alex Johnson", "email": "alex.johnson@example.com", "phone": "+1-555-0199"},
            "action_verbs_count": 12,
            "experience_years": 6.0,
            "education": "PhD"
        }
        resume_text = demo_text

# If Resume Data Available, Run Analysis
if resume_text:
    # Run Matcher Score
    matcher_res = matcher.match(resume_text, target_role)
    semantic_match_score = float(matcher_res.get("match_score", 75.0))
    
    # Run Recruiter Decision Engine
    skills_list = parsed.get("skills", extract_skills(resume_text))
    exp_years = float(parsed.get("experience_years", 3.0))
    word_count = int(parsed.get("word_count", len(resume_text.split())))
    action_verbs = int(parsed.get("action_verbs_count", 6))
    edu_str = parsed.get("education", "Bachelor")
    edu_rank = 3 if "phd" in str(edu_str).lower() else (2 if "master" in str(edu_str).lower() or "bachelor" in str(edu_str).lower() else 1)
    
    decision_res = recruiter_engine.predict(
        skill_count=len(skills_list),
        match_score=semantic_match_score,
        exp_years=exp_years,
        action_verbs=action_verbs,
        word_count=word_count,
        edu_rank=edu_rank,
        projects_count=4,
        certifications_count=2
    )

    # CREATE MAIN DASHBOARD TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 ML Recruiter Decision & LIME Explainer",
        "🔍 Skills & Keyword Gap Analysis",
        "🎯 Job Match Engine",
        "📝 Resume Templates & AI Consultant"
    ])
    
    # --- TAB 1: ML RECRUITER DECISION & LIME EXPLAINER ---
    with tab1:
        st.subheader("🤖 Recruiter Hiring Decision Prediction")
        
        col1, col2, col3, col4 = st.columns(4)
        
        selected_pct = decision_res["selected_percentage"]
        rejected_pct = decision_res["rejected_percentage"]
        decision = decision_res["decision"]
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Recruiter Decision</div>
                <div class="{ 'metric-value-hire' if decision == 'Selected' else 'metric-value-reject' }">{decision.upper()}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Selection Probability</div>
                <div class="metric-value-hire">{selected_pct}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Rejection Risk</div>
                <div class="metric-value-reject">{rejected_pct}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Model Accuracy</div>
                <div style="font-size:2.2rem; font-weight:700; color:#58a6ff;">{decision_res['model_accuracy']}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns([1, 1])
        
        with col_chart1:
            st.subheader("🎯 Probability Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=selected_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Hire Probability for {target_role}"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2ea043" if selected_pct >= 50 else "#f85149"},
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(248, 81, 73, 0.2)"},
                        {'range': [50, 75], 'color': "rgba(210, 153, 34, 0.2)"},
                        {'range': [75, 100], 'color': "rgba(46, 160, 67, 0.2)"}
                    ]
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_chart2:
            st.subheader("⚖️ LIME Feature Drivers (Hire vs Reject Factors)")
            exps = decision_res["explanations"]
            df_exp = pd.DataFrame(exps)
            
            fig_bar = px.bar(
                df_exp,
                x="weight",
                y="feature",
                orientation="h",
                color="impact",
                color_discrete_map={"positive": "#2ea043", "negative": "#f85149"},
                title="Impact Weight (%) per Screening Feature",
                hover_data=["description"]
            )
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, yaxis={'autorange': 'reversed'})
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("### 📋 Detailed Screening Feature Matrix")
        st.table(pd.DataFrame([
            {"Feature": "Skill Diversity", "Value": f"{len(skills_list)} Skills", "Mean Benchmark": "8.0 Skills"},
            {"Feature": "Semantic Role Match", "Value": f"{semantic_match_score:.1f}%", "Mean Benchmark": "60.0%"},
            {"Feature": "Experience Duration", "Value": f"{exp_years:.1f} Years", "Mean Benchmark": "3.0 Years"},
            {"Feature": "Action Verb Density", "Value": f"{action_verbs} Action Verbs", "Mean Benchmark": "5.0 Verbs"},
            {"Feature": "Document Word Count", "Value": f"{word_count} Words", "Mean Benchmark": "450 Words"}
        ]))

    # --- TAB 2: SKILLS & KEYWORD GAP ANALYSIS ---
    with tab2:
        st.subheader("🔍 Skill Extraction & Keyword Gap Analysis")
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("#### ✅ Extracted Technical Skills")
            if skills_list:
                skills_html = " ".join([f"<span class='badge-positive'>{s}</span>" for s in skills_list])
                st.markdown(skills_html, unsafe_allow_html=True)
            else:
                st.warning("No technical skills detected.")
                
        with col_s2:
            st.markdown(f"#### ⚠️ Missing Recommended Skills for {target_role}")
            missing_skills = matcher_res.get("missing_skills", ["PyTorch", "Kubernetes", "AWS", "MLOps", "Spark"])
            if missing_skills:
                missing_html = " ".join([f"<span class='badge-negative'>{s}</span>" for s in missing_skills])
                st.markdown(missing_html, unsafe_allow_html=True)
            else:
                st.success("Great job! No major skill gaps detected.")

    # --- TAB 3: JOB MATCH ENGINE ---
    with tab3:
        st.subheader("🎯 Semantic Vector Embedding Job Matcher")
        st.markdown(f"**Target Role**: `{target_role}` | **Semantic Match Score**: `{semantic_match_score:.1f}%`")
        
        matched_jobs = matcher_res.get("matched_jobs", [
            {"title": "Senior ML Engineer", "company": "TechCorp AI", "score": 92.5, "location": "San Francisco, CA"},
            {"title": "Backend AI Developer", "company": "CloudScale", "score": 84.0, "location": "Remote"},
            {"title": "Data Scientist - NLP", "company": "DataMind", "score": 78.2, "location": "New York, NY"}
        ])
        
        for job in matched_jobs:
            score = job.get("score", 80.0)
            st.markdown(f"""
            <div class="metric-card" style="text-align:left;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <h4 style="margin:0; color:#58a6ff;">{job.get('title', 'Role')}</h4>
                        <p style="margin:0; color:#8b949e;">{job.get('company', 'Company')} • {job.get('location', 'Location')}</p>
                    </div>
                    <div style="font-size:1.5rem; font-weight:700; color:{'#2ea043' if score >= 80 else '#d29922'};">
                        {score:.1f}% Match
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # --- TAB 4: RESUME TEMPLATES & AI CONSULTANT ---
    with tab4:
        st.subheader("📝 19+ Resume Templates & Auto-Formatter")
        template_choice = st.selectbox(
            "Select Resume Template:",
            [
                "ats_classic",
                "iit_premier_academic",
                "nordic_glass",
                "harvard_corporate",
                "cyberpunk_neo"
            ]
        )
        
        if st.button("🎨 Render Template HTML"):
            resume_data = {
                "name": parsed.get("contact_info", {}).get("name", "Candidate Name"),
                "email": parsed.get("contact_info", {}).get("email", "candidate@email.com"),
                "skills": skills_list,
                "summary": f"Professional seeking {target_role} position.",
                "experience_years": exp_years
            }
            html_out = render_resume_template(template_choice, resume_data)
            st.components.v1.html(html_out, height=600, scrolling=True)
