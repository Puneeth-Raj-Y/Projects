"""
engine/recommender.py
---------------------
Ranks a list of processed job dicts by their match_score.
Each job dict must have a "match_score" key (float, 0–100).
"""

from typing import List, Dict, Any


def rank_jobs(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort jobs by match_score in descending order (best match first).

    Args:
        jobs: A list of processed job dicts, each containing at minimum:
              {
                  "title": str,
                  "match_score": float,   # 0 – 100
                  ...
              }

    Returns:
        The same list sorted by "match_score" descending.
        Jobs with equal scores preserve their original relative order (stable sort).

    Example:
        >>> rank_jobs([{"title": "A", "match_score": 40},
        ...            {"title": "B", "match_score": 80}])
        [{"title": "B", "match_score": 80}, {"title": "A", "match_score": 40}]
    """
    return sorted(jobs, key=lambda job: job.get("match_score", 0), reverse=True)


def top_n(jobs: List[Dict[str, Any]], n: int = 10) -> List[Dict[str, Any]]:
    """
    Return the top-n ranked jobs.

    Args:
        jobs: Unsorted or pre-sorted list of job dicts.
        n:    Maximum number of results to return.

    Returns:
        Up to *n* jobs sorted by match_score descending.
    """
    return rank_jobs(jobs)[:n]
