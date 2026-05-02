# 🚀 CareerPulse AI — Career Intelligence Engine

CareerPulse AI is a **Flask-based career intelligence system** that matches user skills with real-time job market data and provides **ranked job recommendations with skill gap analysis**.

This project moves beyond basic job portals by implementing a **decision engine** that analyzes, scores, and explains career opportunities.

---

## 🧠 Core Idea

Instead of just showing jobs, this system answers:

> **“Why is this job relevant to you?”**

It does this using:

* Skill matching
* Scoring algorithms
* Gap analysis
* Real-time job data

---

## ✨ Features

### 🔍 Smart Job Matching

* Fetches real-time jobs using RapidAPI (JSearch)
* Matches user skills with job requirements
* Ranks jobs based on relevance score

### 📊 Skill Gap Analysis

* Identifies missing skills for each role
* Helps users understand what to learn next

### ⚡ Multi-Source Job Fetching (Extensible)

* Supports integration of multiple job APIs
* Fallback system ensures results availability

### 🔐 Authentication System

* User login & registration
* Session-based authentication

### 📱 Modern UI

* Clean dashboard interface
* Responsive design
* Progressive Web App (PWA) support

---

## 🏗️ Project Architecture

```
CareerPulse_AI/
│
├── app.py                 # Flask routes and main application
├── users.db               # SQLite database
├── /engine                # Core logic layer (modular system)
│   ├── skill_extractor.py
│   ├── job_processor.py
│   ├── matcher.py
│   ├── gap_analysis.py
│   └── recommender.py
│
├── /templates            # HTML templates (Jinja2)
├── /static               # CSS, JS, PWA files
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

### Backend

* Python (Flask)
* SQLite (can be upgraded to PostgreSQL/MySQL)

### APIs

* RapidAPI JSearch and RapidAPI Linkedin Job Search (Job Data)

### Frontend

* HTML5, CSS3
* Jinja2 templating
* JavaScript (PWA support)

---

## 🔄 How It Works

1. User inputs:

   * Role
   * Skills
   * Location

2. System:

   * Calls job API
   * Extracts job data
   * Matches skills using scoring logic

3. Output:

   * Ranked job list
   * Match percentage
   * Missing skills per job

---

## 🚀 Getting Started

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd CareerPulse_AI
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Setup Environment Variables

Create a `.env` file:

```env
SECRET_KEY=your_secret_key
RAPID_API_KEY=your_rapidapi_key
```

---

### 4. Run Application

```bash
python app.py
```

Open:

```
http://127.0.0.1:5000
```

---

## 🧪 Example Input

```
Role: Backend Developer
Skills: Python, SQL
Location: India
```

### Output:

* Ranked job listings
* Match score (e.g., 72%)
* Missing skills (e.g., Docker, System Design)

---

## 📈 Future Improvements

* Resume upload & parsing (PDF support)
* Skill demand analytics (market trends)
* Machine learning-based recommendation engine
* Deployment (Docker + Cloud)

---

## ⚠️ Known Limitations

* Depends on external API availability
* Skill matching is rule-based (can be improved with ML)
* API rate limits may affect results

---

## 🧠 Why This Project Stands Out

Unlike typical CRUD job portals, this project demonstrates:

* System design thinking
* API integration
* Data processing & scoring logic
* Real-world problem solving

---

## 👨‍💻 Author

**Puneeth Raj**
MCA Student | Aspiring Software Engineer

---

## 📄 License

This project is for educational and portfolio purposes.
