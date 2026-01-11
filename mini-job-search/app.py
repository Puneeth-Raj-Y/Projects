from flask import Flask, render_template, request, redirect, session
import psycopg2
import requests

app = Flask(__name__)
app.secret_key = "minisearch_secret"

# ---------------- DATABASE ----------------
DB_HOST = "ep-crimson-union-abpx4xvf-pooler.eu-west-2.aws.neon.tech"
DB_NAME = "neondb"
DB_USER = "neondb_owner"
DB_PASS = "npg_KYxM0LSTdig5"
DB_PORT = "5432"

def get_db():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT
    )

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
        c.execute("SELECT id FROM users WHERE username=%s AND password=%s",(u,p))
        user=c.fetchone()
        conn.close()

        if user:
            session["uid"]=user[0]
            return redirect("/dashboard")
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
            c.execute("INSERT INTO users(username,password) VALUES(%s,%s)",(u,p))
            conn.commit()
            msg="success"
        except:
            msg="exists"
        conn.close()

    return render_template("register.html",msg=msg)

# ---------------- DASHBOARD ----------------
@app.route("/dashboard", methods=["GET","POST"])
def dashboard():
    if "uid" not in session:
        return redirect("/login")

    results=[]

    if request.method=="POST":
        role = request.form.get("role","").lower()
        skills = request.form.get("skills","").lower()
        location = request.form.get("location","").lower()
        exp = request.form.get("experience","0")

        url="https://jsearch.p.rapidapi.com/search"

        headers={
            "X-RapidAPI-Key": "43f5091e61mshc8782348e0e6cb7p1c537bjsn506876b76d89",
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
        }

        query = f"{role} {skills} jobs in {location}"

        params = {
            "query": query,
            "page": "1",
            "num_pages": "1"
        }

        response = requests.get(url, headers=headers, params=params)
        data = response.json().get("data", [])

        skills_list=[s.strip() for s in skills.split(",") if s.strip()]

        for job in data:
            title = job.get("job_title","").lower()
            desc = job.get("job_description","").lower()

            score=0
            for s in skills_list:
                if s in title or s in desc:
                    score+=1

            if score>0:
                results.append((
                    job.get("job_title",""),
                    job.get("employer_name",""),
                    f'{job.get("job_city","")} {job.get("job_country","")}',
                    job.get("job_apply_link",""),
                    score
                ))

        results.sort(key=lambda x:x[4], reverse=True)

    return render_template("index.html",results=results)

app.run(debug=True)
