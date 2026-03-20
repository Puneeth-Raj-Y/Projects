import os
import socket
from flask import Flask, render_template, request, redirect, session, url_for
import sqlite3
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = os.getenv("SECRET_KEY", "careerpulse_secret_key_123")

# ---------------- AI SCORING ----------------
def get_ai_match_info(job_title, job_desc, user_skills):
    """Uses OpenAI to get a match score and skill gap analysis."""
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        return None # Fallback to basic scoring

    prompt = f"""
    Compare this job with the user's skills:
    Job Title: {job_title}
    Description: {job_desc[:1000]}
    User Skills: {user_skills}

    Return ONLY a JSON object with:
    1. "score": (0-100) match percentage
    2. "analysis": (1 sentence) why it matches or what is missing.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        import json
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"AI Error: {e}")
        return None

# ---------------- DATABASE ----------------
DB_PATH = os.getenv("DB_PATH", "users.db")

def init_sqlite_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row # Allows dictionary-like access
    return conn

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("home.html")

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET","POST"])
def login():
    msg=""
    if request.method=="POST":
        u=request.form["username"]
        p=request.form["password"]

        conn=get_db()
        c=conn.cursor()
        c.execute("SELECT id FROM users WHERE username=? AND password=?", (u, p))
        user=c.fetchone()
        conn.close()

        if user:
            session["uid"]=user["id"]
            return redirect(url_for("dashboard"))
        else:
            msg="Invalid login"

    return render_template("login.html",msg=msg)

# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET","POST"])
def register():
    msg=""
    if request.method=="POST":
        u=request.form["username"]
        p=request.form["password"]

        conn=get_db()
        c=conn.cursor()
        try:
            c.execute("INSERT INTO users(username,password) VALUES(?,?)", (u, p))
            conn.commit()
            msg="success"
        except sqlite3.IntegrityError:
            msg="exists"
        except Exception as e:
            print(f"Registration Error: {e}")
            msg="error"
        conn.close()

    return render_template("register.html",msg=msg)

# ---------------- PROFILE (Public) ----------------
@app.route("/profile/<int:uid>")
def profile(uid):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE id=?", (uid,))
    user = c.fetchone()
    conn.close()
    
    if not user:
        return "Profile not found", 404
        
    return render_template("profile.html", username=user["username"])

# ---------------- DASHBOARD ----------------
@app.route("/dashboard", methods=["GET","POST"])
def dashboard():
    if "uid" not in session:
        return redirect(url_for("login"))

    results = []
    # Get host IP for Mobile QR access
    local_ip = get_local_ip()
    base_url = f"http://{local_ip}:5000"

    if request.method == "POST":
        role = request.form.get("role", "").lower()
        skills = request.form.get("skills", "").lower()
        location = request.form.get("location", "")
        if not location:
            location = "India"
        
        # Explicit query for Indian jobs
        query = f"{role} {skills} jobs in {location}"
        params = {"query": query, "page": "1", "num_pages": "1"}
        
        url = "https://jsearch.p.rapidapi.com/search"
        headers = {
            "X-RapidAPI-Key": os.getenv("RAPID_API_KEY"),
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
        }

        try:
            print(f"DEBUG: Searching with query: {query}")
            response = requests.get(url, headers=headers, params=params)
            print(f"DEBUG: API Response Status: {response.status_code}")
            data = response.json()
            
            job_list = data.get("data", [])
            print(f"DEBUG: Found {len(job_list)} jobs")
            
            skills_list = [s.strip() for s in skills.split(",") if s.strip()]

            for job in job_list:
                desc = job.get("job_description", "")
                title = job.get("job_title", "")
                
                # Basic Match Score
                match_score = 0
                for s in skills_list:
                    if s.lower() in title.lower() or s.lower() in desc.lower():
                        match_score += 1
                
                # Normalize score for display (0-100)
                display_score = min(100, (match_score / max(1, len(skills_list))) * 100)

                # AI Enrichment (Optional)
                ai_info = get_ai_match_info(job.get("job_title", ""), desc, skills)
                
                final_score = display_score
                analysis = "Matched based on keyword analysis."
                
                if ai_info:
                    final_score = ai_info.get("score", display_score)
                    analysis = ai_info.get("analysis", analysis)

                results.append({
                    "title": job.get("job_title", "N/A"),
                    "employer": job.get("employer_name", "N/A"),
                    "location": f'{job.get("job_city", "")} {job.get("job_country", "")}',
                    "apply_link": job.get("job_apply_link", "#"),
                    "score": round(final_score),
                    "description": analysis or (job.get("job_description", "")[:200] + "...")
                })

            results.sort(key=lambda x: x["score"], reverse=True)
        except Exception as e:
            print(f"Error fetching jobs: {e}")
    else:
        # Default India-centric view on GET
        role = "Software Engineer"
        location = "India"
        skills = "Python, SQL"
        
        url = "https://jsearch.p.rapidapi.com/search"
        headers = {
            "X-RapidAPI-Key": os.getenv("RAPID_API_KEY"),
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
        }
        query = f"{role} in {location}"
        params = {"query": query, "page": "1", "num_pages": "1"}

        try:
            response = requests.get(url, headers=headers, params=params)
            data = response.json().get("data", [])
            for job in data:
                results.append({
                    "title": job.get("job_title", "N/A"),
                    "employer": job.get("employer_name", "N/A"),
                    "location": f'{job.get("job_city", "")} {job.get("job_country", "")}',
                    "apply_link": job.get("job_apply_link", "#"),
                    "score": 70, # Default score for initial view
                    "description": job.get("job_description", "")[:200] + "..."
                })
        except Exception as e:
            print(f"Error fetching default jobs: {e}")

    return render_template("index.html", results=results, base_url=base_url)

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

if __name__ == "__main__":
    init_sqlite_db()
    app.run(debug=True)
