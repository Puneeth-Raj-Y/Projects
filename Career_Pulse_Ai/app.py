"""
app.py — CareerPulse AI
========================
Routes only.  All business logic lives in /engine and /database.
"""

import os
import requests

import time
from flask import Flask, render_template, request, redirect, session, url_for
from dotenv import load_dotenv

from engine import (
    extract_skills,
    process_job,
    calculate_match_score,
    get_skill_gap,
    summarise_gap,
    rank_jobs,
)
from database import init_db, create_user, verify_user, get_user_by_id, upsert_job, save_recommendation

# ---------------------------------------------------------------------------
load_dotenv()
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.secret_key = os.getenv("SECRET_KEY", "careerpulse_secret_key_123")

RAPID_API_KEY = os.getenv("RAPID_API_KEY", "")
JSEARCH_URL   = "https://jsearch.p.rapidapi.com/search"
JSEARCH_HEADERS = {
    "X-RapidAPI-Key":  RAPID_API_KEY,
    "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
}

LINKEDIN_API_KEY = os.getenv("LINKEDIN_API_KEY", RAPID_API_KEY)
LINKEDIN_URL     = "https://linkedin-jobs-search.p.rapidapi.com/"
LINKEDIN_HEADERS = {
    "X-RapidAPI-Key":  LINKEDIN_API_KEY,
    "X-RapidAPI-Host": "linkedin-jobs-search.p.rapidapi.com",
}

# Startup key check
if not RAPID_API_KEY:
    print("[WARNING] RAPID_API_KEY is not set — API calls will be skipped.")
    print("[WARNING] Create a .env file with: RAPID_API_KEY=your_key_here")
else:
    print(f"[INFO] RAPID_API_KEY loaded (ends: ...{RAPID_API_KEY[-6:]})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_search_queries(role: str, location: str) -> list:
    """Build broad and specific search queries for JSearch."""
    role = (role or "").strip()
    location = (location or "").strip()

    # Broaden the role if it's too specific (e.g. "Senior Python Developer" -> "Senior Python")
    base_role = role
    for sep in [",", " with ", " - ", " — ", " – ", "|"]:
        if sep in base_role:
            base_role = base_role.split(sep)[0].strip()

    # Generate a set of query variations
    queries = []
    
    # 1. Specific (Role + Location)
    if base_role and location:
        queries.append(f"{base_role} in {location}")
        queries.append(f"{base_role} {location}")
    
    # 2. Broad (Role only or Location only)
    if base_role:
        queries.append(f"{base_role} jobs")
        queries.append(base_role)
    
    if location:
        queries.append(f"Software Engineer in {location}")
        queries.append(location)

    # Dedup and limit
    deduped = []
    seen = set()
    for q in queries:
        normalized = " ".join(q.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)

    # Ensure we always have at least one valid search query
    return deduped[:5] or ["Software Engineer"]


def normalize_job(job: dict, source: str) -> dict:
    """Normalize job data from different API sources into a unified format."""
    if source == "jsearch":
        return {
            "title":      job.get("job_title", "N/A"),
            "employer":   job.get("employer_name", "N/A"),
            "location":   f"{job.get('job_city', '')}, {job.get('job_country', '')}".strip(", "),
            "description": job.get("job_description", ""),
            "apply_link": job.get("job_apply_link", "#"),
            "raw_skills": [] # Will be populated by process_job later
        }
    elif source == "linkedin":
        # LinkedIn API (RapidAPI) typically returns fields like 'job_title', 'company_name', etc.
        return {
            "title":      job.get("job_title") or job.get("title") or "N/A",
            "employer":   job.get("company_name") or job.get("company") or "N/A",
            "location":   job.get("location") or "N/A",
            "description": job.get("description") or job.get("job_description") or "",
            "apply_link": job.get("job_apply_url") or job.get("url") or "#",
            "raw_skills": []
        }
    return job


def fetch_jobs_jsearch(query: str) -> list:
    """Fetch all possible jobs from JSearch API using pagination."""
    if not RAPID_API_KEY:
        print("[DEBUG] JSearch: No API Key")
        return []
    
    all_jobs = []
    for page in range(1, 4):  # Fetch up to 3 pages
        params = {"query": query, "page": str(page), "num_pages": "1"}
        print(f"[DEBUG] JSearch Request | Query: '{query}' | Page: {page}")
        try:
            # Respect rate limits between pages
            if page > 1:
                time.sleep(1.5)
            resp = requests.get(JSEARCH_URL, headers=JSEARCH_HEADERS, params=params, timeout=10)
            if resp.status_code == 200:
                jobs = resp.json().get("data", [])
                if not jobs:
                    break
                all_jobs.extend(jobs)
                print(f"[DEBUG] JSearch: Found {len(jobs)} jobs on page {page}")
            else:
                print(f"[ERROR] JSearch page {page} failed: {resp.status_code}")
                break
        except Exception as e:
            print(f"[ERROR] JSearch failed: {e}")
            break
            
    print(f"[DEBUG] JSearch: Total jobs fetched: {len(all_jobs)}")
    return all_jobs


def fetch_jobs_linkedin(query: str) -> list:
    """Fetch all possible jobs from LinkedIn API using pagination."""
    if not LINKEDIN_API_KEY:
        print("[DEBUG] LinkedIn: No API Key")
        return []
    
    all_jobs = []
    for page in range(1, 4):  # Fetch up to 3 pages
        params = {"search_terms": query, "location": "India", "page": str(page)}
        print(f"[DEBUG] LinkedIn Request | Query: '{query}' | Page: {page}")
        try:
            if page > 1:
                time.sleep(1.5)
            resp = requests.get(LINKEDIN_URL, headers=LINKEDIN_HEADERS, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                jobs = data if isinstance(data, list) else data.get("data", [])
                if not jobs:
                    break
                all_jobs.extend(jobs)
                print(f"[DEBUG] LinkedIn: Found {len(jobs)} jobs on page {page}")
            else:
                print(f"[ERROR] LinkedIn page {page} failed: {resp.status_code}")
                break
        except Exception as e:
            print(f"[ERROR] LinkedIn failed: {e}")
            break
            
    print(f"[DEBUG] LinkedIn: Total jobs fetched: {len(all_jobs)}")
    return all_jobs


def _fetch_all_jobs(role: str, location: str) -> list:
    """Main job fetcher: calls multiple APIs, merges and dedups results."""
    queries = _build_search_queries(role, location)
    primary_query = queries[0] if queries else f"{role} in {location}"
    
    # 1. Fetch from JSearch
    raw_jsearch = fetch_jobs_jsearch(primary_query)
    print(f"[DEBUG] JSearch results: {len(raw_jsearch)}")
    
    # 2. Fetch from LinkedIn
    raw_linkedin = fetch_jobs_linkedin(primary_query)
    print(f"[DEBUG] LinkedIn results: {len(raw_linkedin)}")
    
    # 3. Normalize and Merge
    combined = []
    for j in raw_jsearch:
        combined.append(normalize_job(j, "jsearch"))
    for j in raw_linkedin:
        combined.append(normalize_job(j, "linkedin"))
        
    # 4. Remove Duplicates (Title + Employer)
    unique_jobs = {}
    for job in combined:
        key = (job["title"] + job["employer"]).lower().strip()
        if key not in unique_jobs:
            unique_jobs[key] = job
            
    final_list = list(unique_jobs.values())
    print(f"[DEBUG] Total unique jobs merged: {len(final_list)}")
    return final_list


def _build_results(normalized_jobs: list, user_skills: list) -> list:
    """
    Pipeline: normalized jobs → processed → scored → gap-analysed.
    Returns a list of dicts ready for the template.
    """
    results = []
    for job_data in normalized_jobs:
        # Reuse process_job logic but handle normalized format
        job = process_job(job_data) 
        score = calculate_match_score(user_skills, job["skills"])
        gap   = get_skill_gap(user_skills, job["skills"])

        results.append({
            "title":        job["title"],
            "employer":     job["employer"],
            "location":     job["location"],
            "apply_link":   job["apply_link"],
            "description":  job["description"],
            "job_skills":   job["skills"],
            "match_score":  round(score),
            "missing_skills": gap,
            "gap_label":    summarise_gap(gap),
        })

    return rank_jobs(results)


# ---------------------------------------------------------------------------
# Routes — public
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("home.html")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    msg = ""
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = verify_user(username, password)
        if user:
            session["uid"]      = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))
        else:
            msg = "Invalid username or password."
    return render_template("login.html", msg=msg)


@app.route("/register", methods=["GET", "POST"])
def register():
    msg = ""
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        msg = create_user(username, password)   # "success" | "exists" | "error"
    return render_template("register.html", msg=msg)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

@app.route("/profile/<int:uid>")
def profile(uid):
    user = get_user_by_id(uid)
    if not user:
        return "Profile not found", 404
    return render_template("profile.html", username=user["username"])


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "uid" not in session:
        return redirect(url_for("login"))

    results    = []
    role       = ""
    skills_raw = ""
    api_error  = ""

    if request.method == "POST":
        role       = request.form.get("role", "").strip()
        skills_raw = request.form.get("skills", "").strip()
        location   = request.form.get("location", "India").strip() or "India"

        # 1. Extract + normalise user skills
        user_skills = extract_skills(skills_raw)
        print(f"[DEBUG] user_skills: {user_skills}")

        # 2. Fetch all jobs (JSearch + LinkedIn)
        merged_jobs = _fetch_all_jobs(role, location)
        
        # 3. Score, gap-analyse, rank
        results = _build_results(merged_jobs, user_skills)
        print(f"[DEBUG] results count: {len(results)}")

        # 4. Persist recommendations to DB
        uid = session["uid"]
        for job_result in results:
            job_id = upsert_job(job_result["title"], job_result["job_skills"])
            save_recommendation(uid, job_id, job_result["match_score"])

    else:
        # Default GET view — show trending jobs from merged sources
        merged_jobs = _fetch_all_jobs("Software Engineer", "India")
        results = _build_results(merged_jobs, [])
        for r in results:
            r["gap_label"] = "Search to see your match"

    username = session.get("username", "User")
    return render_template(
        "index.html",
        results=results,
        username=username,
        role=role,
        skills_raw=skills_raw,
        api_error=api_error
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
