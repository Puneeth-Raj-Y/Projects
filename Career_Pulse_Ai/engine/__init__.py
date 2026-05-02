# engine/__init__.py
# Exposes the CareerPulse AI engine modules as a package.

from .skill_extractor import extract_skills
from .job_processor import process_job
from .matcher import calculate_match_score
from .gap_analysis import get_skill_gap, summarise_gap
from .recommender import rank_jobs, top_n

__all__ = [
    "extract_skills",
    "process_job",
    "calculate_match_score",
    "get_skill_gap",
    "summarise_gap",
    "rank_jobs",
    "top_n",
]
