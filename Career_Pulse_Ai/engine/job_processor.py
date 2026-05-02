"""
engine/job_processor.py
-----------------------
Converts a raw JSearch API job dict into a normalised internal format with
detected skills. Uses a curated skill vocabulary — no external LLM required.
"""

from typing import Dict, List, Any
import re

# ---------------------------------------------------------------------------
# Curated skill vocabulary (extend freely)
# ---------------------------------------------------------------------------
KNOWN_SKILLS: List[str] = [
    # Languages
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go",
    "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "matlab",
    "bash", "shell", "perl", "dart",

    # Web / Frontend
    "html", "css", "react", "angular", "vue", "next.js", "nuxt", "svelte",
    "jquery", "bootstrap", "tailwind", "webpack", "vite", "redux", "graphql",

    # Backend / Frameworks
    "node.js", "express", "django", "flask", "fastapi", "spring", "spring boot",
    "rails", "laravel", "asp.net", "nestjs",

    # Data & ML
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
    "matplotlib", "seaborn", "hugging face", "llm", "langchain",
    "data analysis", "data science", "feature engineering", "etl",

    # Databases
    "sql", "mysql", "postgresql", "sqlite", "mongodb", "redis", "cassandra",
    "elasticsearch", "dynamodb", "oracle", "firebase",

    # Cloud & DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "terraform",
    "ansible", "jenkins", "github actions", "ci/cd", "linux", "nginx",
    "prometheus", "grafana", "helm",

    # Data Engineering / BI
    "spark", "hadoop", "kafka", "airflow", "dbt", "snowflake", "bigquery",
    "redshift", "tableau", "power bi", "looker", "excel",

    # General Engineering & Architecture
    "rest api", "microservices", "system design", "agile", "scrum",
    "git", "jira", "confluence", "figma", "postman", "swagger",

    # Security
    "cybersecurity", "penetration testing", "owasp", "siem", "oauth",

    # Mobile
    "android", "ios", "react native", "flutter",
]

# Pre-compile patterns for performance
_SKILL_PATTERNS: List[tuple] = [
    (skill, re.compile(r"\b" + re.escape(skill) + r"\b", re.IGNORECASE))
    for skill in KNOWN_SKILLS
]


def process_job(job_json: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a job dict. Supports both raw JSearch keys and normalized keys."""
    title: str = job_json.get("job_title") or job_json.get("title") or "N/A"
    description: str = job_json.get("job_description") or job_json.get("description") or ""
    employer: str = job_json.get("employer_name") or job_json.get("employer") or "N/A"
    
    # Handle location from different sources
    city = job_json.get("job_city", "")
    country = job_json.get("job_country", "")
    location = job_json.get("location") or f"{city} {country}".strip() or "N/A"
    
    apply_link: str = job_json.get("job_apply_link") or job_json.get("apply_link") or "#"

    # Detect skills from combined title + description text
    search_corpus = f"{title} {description}"
    detected_skills = _detect_skills(search_corpus)

    return {
        "title": title,
        "description": description[:500] + ("..." if len(description) > 500 else ""),
        "skills": detected_skills,
        "employer": employer,
        "location": location,
        "apply_link": apply_link,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_skills(text: str) -> List[str]:
    """Return a sorted list of known skills found in *text*."""
    found = []
    for skill, pattern in _SKILL_PATTERNS:
        if pattern.search(text):
            found.append(skill)
    return sorted(set(found))
