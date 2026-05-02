"""
engine/matcher.py
-----------------
Computes a match score between a user's skill set and a job's required skills.

Two strategies are available:
    1. Intersection ratio  (default, no dependencies)
    2. Cosine similarity   (requires scikit-learn; falls back gracefully)

The public API is a single function: calculate_match_score().
"""

from typing import List
import math


def calculate_match_score(
    user_skills: List[str],
    job_skills: List[str],
    strategy: str = "intersection",
) -> float:
    """
    Calculate how well the user's skills match the job requirements.

    Args:
        user_skills: Normalised list of user skill strings (lowercase).
        job_skills:  Normalised list of skills extracted from the job posting.
        strategy:    "intersection" (default) or "cosine".

    Returns:
        A float in [0.0, 100.0] representing the match percentage.
        Returns 0.0 if job_skills is empty.

    Formula (intersection strategy):
        score = (|user_skills ∩ job_skills| / |job_skills|) * 100

    Example:
        >>> calculate_match_score(["python", "sql"], ["python", "sql", "docker"])
        66.67
    """
    if not job_skills:
        return 0.0

    if strategy == "cosine":
        return _cosine_score(user_skills, job_skills)

    return _intersection_score(user_skills, job_skills)


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _intersection_score(user_skills: List[str], job_skills: List[str]) -> float:
    """Simple overlap ratio: common skills / total job skills × 100."""
    user_set = set(user_skills)
    job_set = set(job_skills)
    common = user_set & job_set
    score = (len(common) / len(job_set)) * 100
    return round(score, 2)


def _cosine_score(user_skills: List[str], job_skills: List[str]) -> float:
    """
    Cosine similarity over a binary skill vocabulary vector.

    Falls back to intersection ratio if an import error occurs.
    """
    try:
        # Build vocabulary
        vocab = sorted(set(user_skills) | set(job_skills))
        if not vocab:
            return 0.0

        user_vec = [1 if s in user_skills else 0 for s in vocab]
        job_vec = [1 if s in job_skills else 0 for s in vocab]

        dot = sum(a * b for a, b in zip(user_vec, job_vec))
        mag_user = math.sqrt(sum(a * a for a in user_vec))
        mag_job = math.sqrt(sum(b * b for b in job_vec))

        if mag_user == 0 or mag_job == 0:
            return 0.0

        similarity = dot / (mag_user * mag_job)
        return round(similarity * 100, 2)

    except Exception:
        # Graceful fallback
        return _intersection_score(user_skills, job_skills)
