"""
ai_integrator.py
AI Integration Service supporting OpenAI (GPT-4o-mini), Anthropic, and local rule-based fallback engines.

Provides:
  - Executive Resume Review & Improvement Suggestions
  - Missing High-Impact Keyword Recommendations
  - AI Bullet Point Rewriter
  - Interactive AI Resume Consultant Chatbot
"""

import os
import json

DEFAULT_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")



def get_ai_suggestions(candidate_data: dict, target_job: dict = None, provider: str = "auto", api_key: str = None) -> dict:
    """
    Generate AI recommendations using OpenAI, Anthropic, or local fallback.
    """
    key_openai = api_key or DEFAULT_OPENAI_KEY
    key_anthropic = api_key or os.getenv("ANTHROPIC_API_KEY")

    if provider in ["openai", "auto"] and key_openai:
        try:
            return _call_openai(candidate_data, target_job, key_openai)
        except Exception as e:
            print(f"OpenAI call failed, falling back to local engine: {e}")

    if provider in ["anthropic", "auto"] and key_anthropic:
        try:
            return _call_anthropic(candidate_data, target_job, key_anthropic)
        except Exception as e:
            print(f"Anthropic call failed, falling back to local engine: {e}")

    # Fallback to local rule-based AI engine
    return _local_rule_engine(candidate_data, target_job)


def chat_with_ai(user_message: str, chat_history: list = None, candidate_data: dict = None, target_job: dict = None, api_key: str = None) -> dict:
    """
    Real-time interactive Chatbot assistant for resume optimization & interview prep.
    """
    key_openai = api_key or DEFAULT_OPENAI_KEY
    candidate_data = candidate_data or {}
    target_job = target_job or {}
    chat_history = chat_history or []

    system_prompt = f"""
You are an expert AI Resume Consultant & Hiring Strategist.
Candidate Profile:
- Name: {candidate_data.get('name', 'Candidate')}
- Skills Extracted: {', '.join(candidate_data.get('skills', []))}
- Experience Years: ~{candidate_data.get('experience_years', '2.5')} years
- Target Role: {target_job.get('title', 'Software / AI Engineer')}

Provide highly actionable, concise, and encouraging advice to help the candidate optimize their resume, pass ATS screening, and impress recruiters. Keep responses well-formatted with bullet points when applicable.
"""

    if key_openai:
        try:
            import httpx
            from openai import OpenAI
            # Clear any proxy env vars that trigger httpx/openai client mismatch
            for p in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
                os.environ.pop(p, None)

            client = OpenAI(api_key=key_openai, http_client=httpx.Client())

            messages = [{"role": "system", "content": system_prompt}]
            for msg in chat_history[-6:]:  # Keep recent context window
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

            messages.append({"role": "user", "content": user_message})

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=600
            )

            reply = response.choices[0].message.content
            return {"reply": reply, "provider": "OpenAI (GPT-4o-mini)"}

        except Exception as e:
            print(f"OpenAI Chat error: {e}")


    # Local Intelligent Fallback Chatbot
    msg_lower = user_message.lower()
    skills = candidate_data.get("skills", ["Python", "Flask", "SQL"])
    target_title = target_job.get("title", "Technical Position")

    if "keyword" in msg_lower or "ats" in msg_lower:
        reply = f"To maximize your ATS match for **{target_title}**, ensure you feature these top keywords: **{', '.join(skills[:5])}**, plus cloud platforms (AWS/GCP), CI/CD, and System Design."
    elif "bullet" in msg_lower or "rewrite" in msg_lower or "experience" in msg_lower:
        reply = f"Here is a high-impact bullet rewrite for your resume:\n• *'Architected end-to-end machine learning and REST backend services using {skills[0] if skills else 'Python'}, reducing latency by 35% across production workloads.'*"
    elif "improve" in msg_lower or "score" in msg_lower:
        reply = f"To boost your Recruiter Selection score:\n1. Quantify achievements with metrics (%, $, scale).\n2. Add strong action verbs like *Spearheaded*, *Architected*, and *Optimized*.\n3. Keep document length between 400–800 words."
    else:
        reply = f"Great question! Based on your profile with expertise in {', '.join(skills[:4])}, I recommend tailoring your summary header directly for {target_title} and highlighting project outcomes."

    return {"reply": reply, "provider": "AI Assistant (Local Rule Engine)"}


def _call_openai(candidate_data: dict, target_job: dict, api_key: str) -> dict:
    import httpx
    from openai import OpenAI
    for p in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
        os.environ.pop(p, None)

    client = OpenAI(api_key=api_key, http_client=httpx.Client())

    prompt = f"""
You are an expert ATS & AI Recruiter consultant. Analyze this candidate resume data against target role.
Candidate Name: {candidate_data.get('name')}
Skills: {', '.join(candidate_data.get('skills', []))}
Target Job Title: {target_job.get('title') if target_job else 'Software Engineer'}
Target Job Description: {target_job.get('description') if target_job else 'N/A'}

Return a JSON object with:
1. "summary": Executive 2-sentence summary of the candidate profile.
2. "improvements": Array of 3-4 actionable feedback bullet points to improve the resume.
3. "suggested_keywords": Array of 6-8 missing high-impact keywords to add for target ATS ranking.
4. "bullet_rewrites": Array of 2 sample rewritten high-impact bullet points with metrics.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    return json.loads(content)



def _call_anthropic(candidate_data: dict, target_job: dict, api_key: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""
Analyze candidate resume data against target job and return JSON only:
Candidate Name: {candidate_data.get('name')}
Skills: {', '.join(candidate_data.get('skills', []))}
Target Job: {target_job.get('title') if target_job else 'Software Engineer'}

Respond ONLY with valid JSON containing:
{{
  "summary": "...",
  "improvements": ["...", "..."],
  "suggested_keywords": ["...", "..."],
  "bullet_rewrites": ["...", "..."]
}}
"""

    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(message.content[0].text)


def _local_rule_engine(candidate_data: dict, target_job: dict) -> dict:
    skills = candidate_data.get("skills", [])
    target_title = target_job.get("title", "Technical Role") if target_job else "Target Role"
    req_skills = target_job.get("required_skills", []) if target_job else ["Python", "Docker", "REST API", "Git", "CI/CD", "AWS"]

    missing_keywords = [s for s in req_skills if s.lower() not in [sk.lower() for sk in skills]]
    if not missing_keywords:
        missing_keywords = ["System Design", "Unit Testing", "CI/CD Pipelines", "Cloud Deployment (AWS/GCP)", "Agile / Scrum"]

    improvements = []
    if len(skills) < 6:
        improvements.append("Expand technical skills section with specific frameworks and tools used in recent projects.")
    else:
        improvements.append("Categorize detected skills into clear sections (e.g., Languages, Frameworks, Cloud & Tools).")

    improvements.append("Quantify achievements in bullet points (e.g., 'Improved API response time by 35%' instead of 'Worked on API').")
    improvements.append(f"Align resume terminology directly with target role keywords for {target_title}.")
    improvements.append("Include links to active GitHub projects, live demos, or portfolio achievements in the header.")

    summary = f"Candidate displays strong potential for {target_title} with foundational skills in {', '.join(skills[:3]) if skills else 'core technologies'}. Optimizing ATS keyword alignment will significantly boost recruiter callbacks."

    bullet_rewrites = [
        "Spearheaded development of scalable REST microservices, reducing latency by 40% across 50k daily active users.",
        "Architected automated CI/CD deployment pipelines using Docker, accelerating feature delivery release cycles by 25%."
    ]

    return {
        "summary": summary,
        "improvements": improvements,
        "suggested_keywords": missing_keywords,
        "bullet_rewrites": bullet_rewrites,
        "provider_used": "Local Rule-based AI Engine"
    }
