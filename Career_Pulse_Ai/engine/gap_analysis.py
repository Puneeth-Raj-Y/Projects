"""
engine/gap_analysis.py
----------------------
Computes the skill gap between a user's current skills and a job's requirements.
Pure Python, zero external dependencies.
"""

from typing import List


def get_skill_gap(user_skills: List[str], job_skills: List[str]) -> List[str]:
    """
    Return skills required by the job that the user does not currently have.

    Args:
        user_skills: Normalised list of user skill strings (lowercase).
        job_skills:  Normalised list of skills extracted from the job posting.

    Returns:
        A sorted list of missing skill strings.
        Returns an empty list if the user meets all requirements.

    Formula:
        missing = job_skills - user_skills

    Example:
        >>> get_skill_gap(["python", "sql"], ["python", "sql", "docker", "kubernetes"])
        ['docker', 'kubernetes']
    """
    user_set = set(user_skills)
    job_set = set(job_skills)
    missing = sorted(job_set - user_set)
    return missing


def summarise_gap(missing_skills: List[str], max_display: int = 5) -> str:
    """
    Return a human-readable summary string of missing skills.

    Args:
        missing_skills: Output of get_skill_gap().
        max_display:    Maximum number of skills to name explicitly.

    Returns:
        A concise display string, e.g. "Docker, Kubernetes (+2 more)"
        or "None — great match! 🎉"
    """
    if not missing_skills:
        return "None — great match! 🎉"

    displayed = missing_skills[:max_display]
    remainder = len(missing_skills) - max_display

    label = ", ".join(s.title() for s in displayed)
    if remainder > 0:
        label += f" (+{remainder} more)"
    return label
