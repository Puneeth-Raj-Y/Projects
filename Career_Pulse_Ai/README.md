# 🚀 CareerPulse AI: Intelligent Job Discovery Platform

**CareerPulse AI** is a professional, AI-powered job search portal designed to bridge the gap between candidate skills and market opportunities. It leverages advanced NLP to provide real-time job matching scores and personalized skill-gap analysis.

## ✨ Key Features

- **📱 Progressive Web App (PWA)**: Install the platform on your mobile device for a native-like experience with offline access.

- **🤖 AI-Powered Match Scoring**: Uses OpenAI's GPT models to analyze job descriptions and calculate a precise "Match Percentage" based on your unique skill set.

## 🛠️ Technology Stack

- **Backend**: Python 3.x, Flask
- **Database**: PostgreSQL (Prisma/Neon DB)
- **AI/ML**: OpenAI GPT API
- **Frontend**: HTML5, Vanilla CSS3 (Custom Design System), Jinja2 Templates
- **API Integration**: RapidAPI (JSearch)
- **Environment Management**: `python-dotenv`

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone [repository-url]
   cd mini-job-search
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory with the following:
   ```env
   SECRET_KEY=your_secret_key
   DB_HOST=your_db_host
   DB_NAME=your_db_name
   DB_USER=your_db_user
   DB_PASS=your_db_password
   DB_PORT=5432
   RAPID_API_KEY=your_rapidapi_key
   OPENAI_API_KEY=your_openai_api_key
   ```

4. **Run the Application**:
   ```bash
   python app.py
   ```

## 📈 Future Roadmap

- [ ] Save Jobs to a personal "Watchlist".
- [ ] Resume Builder with AI-driven content suggestions.
- [ ] Direct application portal integration.

---
*Developed by Puneeth Raj | MCA Project*
