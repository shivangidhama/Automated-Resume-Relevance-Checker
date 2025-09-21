# 🎯 Automated Resume Relevance Check System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://automated-resume-relevance-checker.streamlit.app/)

**Hackathon:** Code4EdTech Hackathon by Innomatics Research Labs  
**Theme 2:** Automated Resume Relevance Check System  

An AI-powered solution for **resume–job description matching**.  
This system helps recruiters and job seekers by automatically analyzing resumes, generating insights, and presenting results in an **interactive dashboard** for efficient recruitment.

---

## 🚀 Live Demo
🔗 [Automated Resume Relevance Checker](https://automated-resume-relevance-checker.streamlit.app/)

---

## 📌 Features
- 📂 **Upload Job Description** – Add single/multiple job descriptions for evaluation.  
- 📑 **Analyze Resume** – Upload resumes in PDF/DOCX format for analysis.  
- 📊 **Resume Insights** – Extracts and highlights key skills, strengths, and gaps.  
- 📈 **Result Analysis Dashboard** – Interactive dashboard showing statistics and trends.  
- 🎯 **Percentage Matches** – AI-powered relevance score between job description and resume.  
- 📤 **Bulk Analysis** – Upload multiple resumes in bulk for batch evaluation.  
- 📊 **Interactive Visuals** – Candidate distribution, score distribution, and evaluation summary.  

---

## 🖥️ Tech Stack
- **Frontend:** Streamlit  
- **Backend:** Python  
- **NLP:** spaCy, scikit-learn, sentence-transformers  
- **Document Parsing:** PyPDF2, docx2txt  
- **Visualization:** Plotly, Matplotlib, Streamlit Components  

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+  
- Virtual environment (recommended)

### Steps
1. *Clone the repo:*
   ```bash
   git clone https://github.com/shivangidhama/Automated-Resume-Relevance-Checker.git
   cd Automated-Resume-Relevance-Checker
   ```
2.*Create and activate a virtual environment:*
  ```bash
  #On macOS/Linux
   python -m venv venv
  source venv/bin/activate

   # On Windows
  python -m venv venv
  venv\Scripts\activate
```

3.*Install dependencies:*
```bash
pip install -r requirements.txt
 ```

4.*Download the spaCy English model (first-time setup):*
```bash
python -m spacy download en_core_web_sm
````
5.*Run the Streamlit app:*
```bash
streamlit run main.py
```





