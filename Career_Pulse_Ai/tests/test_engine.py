"""
tests/test_engine.py
--------------------
Unit tests for all five engine modules.
Run with:  python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from engine.skill_extractor import extract_skills
from engine.job_processor   import process_job, _detect_skills
from engine.matcher         import calculate_match_score, _intersection_score, _cosine_score
from engine.gap_analysis    import get_skill_gap, summarise_gap
from engine.recommender     import rank_jobs, top_n


# ═══════════════════════════════════════════════════════════
# MODULE 1 — skill_extractor
# ═══════════════════════════════════════════════════════════

class TestExtractSkills:
    def test_basic_extraction(self):
        result = extract_skills("Python, SQL, Machine Learning")
        assert "python" in result
        assert "sql" in result
        assert "machine learning" in result

    def test_deduplication(self):
        result = extract_skills("sql, SQL, Sql")
        assert result.count("sql") == 1

    def test_whitespace_trimming(self):
        result = extract_skills("  python ,   django  ")
        assert "python" in result
        assert "django" in result

    def test_empty_string(self):
        assert extract_skills("") == []

    def test_none_input(self):
        assert extract_skills(None) == []

    def test_sorted_output(self):
        result = extract_skills("React, Angular, Vue")
        assert result == sorted(result)

    def test_single_skill(self):
        result = extract_skills("docker")
        assert result == ["docker"]

    def test_filters_empty_tokens(self):
        result = extract_skills(",,,python,,")
        assert result == ["python"]


# ═══════════════════════════════════════════════════════════
# MODULE 2 — job_processor
# ═══════════════════════════════════════════════════════════

SAMPLE_JOB = {
    "job_title": "Senior Python Developer",
    "job_description": (
        "We need a developer with Python, Django, and PostgreSQL experience. "
        "Docker knowledge is a plus. CI/CD pipeline experience preferred."
    ),
    "employer_name": "TechCorp",
    "job_city": "Bangalore",
    "job_country": "India",
    "job_apply_link": "https://example.com/apply",
}

class TestProcessJob:
    def test_output_keys(self):
        result = process_job(SAMPLE_JOB)
        assert set(result.keys()) >= {"title", "description", "skills", "employer", "location", "apply_link"}

    def test_title_extracted(self):
        result = process_job(SAMPLE_JOB)
        assert result["title"] == "Senior Python Developer"

    def test_skills_detected(self):
        result = process_job(SAMPLE_JOB)
        assert "python" in result["skills"]
        assert "django" in result["skills"]

    def test_description_truncated(self):
        long_desc = "python " * 200
        job = {**SAMPLE_JOB, "job_description": long_desc}
        result = process_job(job)
        assert len(result["description"]) <= 503  # 500 + "..."

    def test_empty_job(self):
        result = process_job({})
        assert result["title"] == "N/A"
        assert result["skills"] == []

    def test_detect_skills_helper(self):
        found = _detect_skills("Experience with React and Node.js required. AWS preferred.")
        assert "react" in found
        assert "node.js" in found
        assert "aws" in found


# ═══════════════════════════════════════════════════════════
# MODULE 3 — matcher
# ═══════════════════════════════════════════════════════════

class TestMatcher:
    def test_perfect_match(self):
        score = calculate_match_score(["python", "sql"], ["python", "sql"])
        assert score == 100.0

    def test_zero_match(self):
        score = calculate_match_score(["excel"], ["python", "kubernetes"])
        assert score == 0.0

    def test_partial_match(self):
        score = calculate_match_score(["python", "sql"], ["python", "sql", "docker"])
        assert round(score, 2) == 66.67

    def test_empty_job_skills(self):
        score = calculate_match_score(["python"], [])
        assert score == 0.0

    def test_empty_user_skills(self):
        score = calculate_match_score([], ["python", "sql"])
        assert score == 0.0

    def test_score_bounded(self):
        score = calculate_match_score(
            ["python", "sql", "docker", "kubernetes", "aws"],
            ["python", "sql"]
        )
        assert 0.0 <= score <= 100.0

    def test_cosine_strategy(self):
        score = calculate_match_score(["python", "sql"], ["python", "sql"], strategy="cosine")
        assert score == 100.0

    def test_intersection_helper(self):
        assert _intersection_score(["a", "b"], ["a", "b", "c"]) == pytest.approx(66.67, abs=0.01)

    def test_cosine_helper_partial(self):
        score = _cosine_score(["python"], ["python", "sql"])
        assert 0 < score < 100


# ═══════════════════════════════════════════════════════════
# MODULE 4 — gap_analysis
# ═══════════════════════════════════════════════════════════

class TestGapAnalysis:
    def test_full_gap(self):
        gap = get_skill_gap([], ["python", "sql", "docker"])
        assert set(gap) == {"python", "sql", "docker"}

    def test_no_gap(self):
        gap = get_skill_gap(["python", "sql"], ["python", "sql"])
        assert gap == []

    def test_partial_gap(self):
        gap = get_skill_gap(["python", "sql"], ["python", "sql", "docker"])
        assert gap == ["docker"]

    def test_gap_is_sorted(self):
        gap = get_skill_gap([], ["sql", "python", "docker"])
        assert gap == sorted(gap)

    def test_summarise_none(self):
        assert "great match" in summarise_gap([])

    def test_summarise_short(self):
        summary = summarise_gap(["docker", "kubernetes"])
        assert "Docker" in summary
        assert "Kubernetes" in summary

    def test_summarise_overflow(self):
        many = [f"skill{i}" for i in range(10)]
        summary = summarise_gap(many, max_display=3)
        assert "+7 more" in summary


# ═══════════════════════════════════════════════════════════
# MODULE 5 — recommender
# ═══════════════════════════════════════════════════════════

JOBS_FIXTURE = [
    {"title": "Junior Dev",    "match_score": 35.0},
    {"title": "Senior Dev",    "match_score": 82.0},
    {"title": "Mid-level Dev", "match_score": 60.0},
]

class TestRecommender:
    def test_ranking_order(self):
        ranked = rank_jobs(JOBS_FIXTURE)
        scores = [j["match_score"] for j in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_top_n(self):
        result = top_n(JOBS_FIXTURE, n=2)
        assert len(result) == 2
        assert result[0]["title"] == "Senior Dev"

    def test_empty_list(self):
        assert rank_jobs([]) == []

    def test_single_item(self):
        jobs = [{"title": "A", "match_score": 50}]
        assert rank_jobs(jobs) == jobs

    def test_missing_score_key(self):
        # Jobs without match_score should default to 0
        jobs = [{"title": "A"}, {"title": "B", "match_score": 70}]
        ranked = rank_jobs(jobs)
        assert ranked[0]["title"] == "B"
