"""
engine/skill_extractor.py
-------------------------
Extracts and normalises a list of skills from raw text input.
No external dependencies — pure Python.
"""

from typing import List


def extract_skills(raw_text: str) -> List[str]:
    """
    Parse a comma-separated skills string into a clean, deduplicated list.

    Args:
        raw_text: A raw string of skills, e.g. "Python, SQL, Machine Learning, sql"

    Returns:
        A sorted list of unique, lowercase, whitespace-stripped skill strings.
        Empty strings are excluded.

    Example:
        >>> extract_skills("Python, SQL, Machine Learning, sql")
        ['machine learning', 'python', 'sql']
    """
    if not raw_text or not isinstance(raw_text, str):
        return []

    skills = [s.strip().lower() for s in raw_text.split(",")]
    # Remove empty tokens and deduplicate while preserving deterministic order
    seen: set = set()
    unique: List[str] = []
    for skill in skills:
        if skill and skill not in seen:
            seen.add(skill)
            unique.append(skill)

    return sorted(unique)
