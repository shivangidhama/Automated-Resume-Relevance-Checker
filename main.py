import os
import re
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import tempfile

# Core libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Resume parsing
import PyPDF2
import docx2txt

# NLP and ML
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Configuration
st.set_page_config(
    page_title="Resume Relevance Check System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric > div > div > div > div {
        font-size: 1.2rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        color: #155724;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        color: #856404;
    }
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeParser:
    """Extract and parse content from resume files"""
    
    def __init__(self):
        self.nlp = None
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            st.warning("⚠️ spaCy model not found. Some features may be limited. Install with: `python -m spacy download en_core_web_sm`")
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting PDF text: {e}")
            return ""
    
    def extract_text_from_docx(self, docx_file) -> str:
        """Extract text from uploaded DOCX file"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                tmp_file.write(docx_file.read())
                tmp_file_path = tmp_file.name
            
            text = docx2txt.process(tmp_file_path)
            os.unlink(tmp_file_path)  # Clean up
            return text
        except Exception as e:
            st.error(f"Error extracting DOCX text: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        text = text.strip()
        
        # Remove common header/footer patterns
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'Confidential.*?Resume', '', text, flags=re.IGNORECASE)
        
        return text
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text using pattern matching"""
        if not text:
            return []
        
        # Common skill patterns
        skill_patterns = [
            r'Skills?:?\s*([^\n]+)',
            r'Technical Skills?:?\s*([^\n]+)',
            r'Programming Languages?:?\s*([^\n]+)',
            r'Technologies?:?\s*([^\n]+)',
            r'Tools?:?\s*([^\n]+)'
        ]
        
        skills = []
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Split by common delimiters
                skill_list = re.split(r'[,;|•\-\n]', match)
                skills.extend([skill.strip() for skill in skill_list if skill.strip() and len(skill.strip()) > 1])
        
        # Extended skills database for matching
        common_skills = [
            'Python', 'Java', 'JavaScript', 'TypeScript', 'React', 'Angular', 'Vue.js',
            'Node.js', 'Express.js', 'Django', 'Flask', 'FastAPI', 'Spring Boot',
            'SQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Elasticsearch',
            'Machine Learning', 'Deep Learning', 'Data Science', 'Pandas', 'NumPy',
            'TensorFlow', 'PyTorch', 'Scikit-learn', 'Keras', 'OpenCV',
            'AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes', 'Jenkins',
            'Git', 'GitHub', 'GitLab', 'CI/CD', 'DevOps', 'Linux', 'Bash',
            'HTML', 'CSS', 'Bootstrap', 'Tailwind CSS', 'SASS', 'LESS',
            'REST API', 'GraphQL', 'Microservices', 'API Development',
            'Unity', 'Unreal Engine', 'Game Development', 'Mobile Development',
            'Android', 'iOS', 'React Native', 'Flutter', 'Xamarin',
            'Business Intelligence', 'Power BI', 'Tableau', 'Excel',
            'Agile', 'Scrum', 'JIRA', 'Project Management',
            'Selenium', 'Testing', 'Unit Testing', 'Integration Testing'
        ]
        
        # Find skills mentioned in text (case-insensitive)
        found_skills = []
        text_lower = text.lower()
        for skill in common_skills:
            if skill.lower() in text_lower:
                found_skills.append(skill)
        
        # Combine and remove duplicates
        all_skills = list(set(skills + found_skills))
        return [skill for skill in all_skills if len(skill) > 2]  # Filter very short skills
    
    def extract_education(self, text: str) -> List[str]:
        """Extract education information"""
        if not text:
            return []
        
        education_patterns = [
            r'(Bachelor.*?(?:Science|Arts|Technology|Engineering|Computer|Information).*?)(?=\n|$)',
            r'(Master.*?(?:Science|Arts|Technology|Engineering|Computer|Information).*?)(?=\n|$)',
            r'(B\.?Tech.*?)(?=\n|$)',
            r'(M\.?Tech.*?)(?=\n|$)',
            r'(MBA.*?)(?=\n|$)',
            r'(PhD.*?)(?=\n|$)',
            r'(University.*?)(?=\n|$)',
            r'(College.*?)(?=\n|$)',
            r'(Institute.*?)(?=\n|$)'
        ]
        
        education = []
        for pattern in education_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            education.extend([match.strip() for match in matches if len(match.strip()) > 5])
        
        return list(set(education))[:5]  # Limit to 5 entries
    
    def extract_experience(self, text: str) -> str:
        """Extract years of experience"""
        if not text:
            return "Not specified"
        
        exp_patterns = [
            r'(\d+[\+\-\s]*(?:to|\-)\s*\d*\s*years?).*?(?:experience|exp)',
            r'(\d+\+?\s*years?).*?(?:experience|exp)',
            r'(?:experience|exp).*?(\d+\+?\s*years?)',
            r'(fresher|fresh graduate|entry.level|no.experience)',
        ]
        
        for pattern in exp_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Not specified"
    
    def parse_resume(self, uploaded_file) -> Dict:
        """Main method to parse resume and extract structured information"""
        if uploaded_file is None:
            return {}
        
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Determine file type and extract text
        file_name = uploaded_file.name.lower()
        if file_name.endswith('.pdf'):
            raw_text = self.extract_text_from_pdf(uploaded_file)
        elif file_name.endswith('.docx'):
            raw_text = self.extract_text_from_docx(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload PDF or DOCX files.")
            return {}
        
        if not raw_text.strip():
            st.error("Could not extract text from the file. Please check if the file is valid.")
            return {}
        
        # Clean text
        clean_text = self.clean_text(raw_text)
        
        # Extract structured information
        skills = self.extract_skills(clean_text)
        education = self.extract_education(clean_text)
        experience = self.extract_experience(clean_text)
        
        return {
            'raw_text': raw_text,
            'clean_text': clean_text,
            'skills': skills,
            'education': education,
            'experience': experience,
            'file_name': uploaded_file.name
        }

class JobDescriptionParser:
    """Parse and structure job descriptions"""
    
    def parse_job_description(self, jd_text: str) -> Dict:
        """Parse job description and extract requirements"""
        if not jd_text.strip():
            return {}
        
        jd_text = jd_text.strip()
        
        # Extract job title (first non-empty line or specific patterns)
        lines = [line.strip() for line in jd_text.split('\n') if line.strip()]
        title = lines[0] if lines else "Unknown Position"
        
        # Try to find more specific title patterns
        title_patterns = [
            r'(?:Job Title|Position|Role):\s*([^\n]+)',
            r'(?:We are looking for|Hiring).*?([A-Z][^\n,]+?)(?:\s|$)',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, jd_text, re.IGNORECASE)
            if match:
                extracted_title = match.group(1).strip()
                if len(extracted_title) < 100:  # Reasonable title length
                    title = extracted_title
                break
        
        # Extract required skills
        required_skills = self.extract_skills_from_jd(jd_text, required=True)
        preferred_skills = self.extract_skills_from_jd(jd_text, required=False)
        
        # Extract experience requirements
        experience = self.extract_experience(jd_text)
        
        # Extract education requirements
        education = self.extract_education_requirements(jd_text)
        
        # Extract location if mentioned
        location = self.extract_location(jd_text)
        
        return {
            'title': title[:200],  # Limit title length
            'raw_text': jd_text,
            'required_skills': required_skills,
            'preferred_skills': preferred_skills,
            'experience': experience,
            'education': education,
            'location': location
        }
    
    def extract_skills_from_jd(self, text: str, required: bool = True) -> List[str]:
        """Extract skills from job description"""
        if required:
            patterns = [
                r'(?:Required|Must have|Essential|Mandatory).*?(?:Skills?|Technologies?|Tools?).*?:?\s*([^\.]+?)(?=\n\n|\n[A-Z]|\n•|\n-|$)',
                r'Requirements?:?\s*([^\.]+?)(?=\n\n|\n[A-Z]|$)',
                r'You must have.*?:?\s*([^\.]+?)(?=\n\n|\n[A-Z]|$)',
            ]
        else:
            patterns = [
                r'(?:Preferred|Nice to have|Good to have|Plus|Bonus|Additional).*?(?:Skills?|Technologies?|Tools?).*?:?\s*([^\.]+?)(?=\n\n|\n[A-Z]|\n•|\n-|$)',
                r'(?:Would be (?:great|nice)|Advantage).*?:?\s*([^\.]+?)(?=\n\n|\n[A-Z]|$)',
            ]
        
        skills = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Clean and split the match
                match = re.sub(r'[•\-]', ',', match)  # Replace bullets and dashes with commas
                skill_list = re.split(r'[,;|\n]', match)
                for skill in skill_list:
                    skill = skill.strip().strip('.')
                    if skill and len(skill) > 2 and len(skill) < 50:
                        skills.append(skill)
        
        # Remove duplicates and clean
        return list(set([skill for skill in skills if skill and len(skill.strip()) > 2]))[:20]  # Limit to 20 skills
    
    def extract_experience(self, text: str) -> str:
        """Extract experience requirements"""
        exp_patterns = [
            r'(\d+[\+\-\s]*(?:to|\-)\s*\d*\s*years?).*?(?:experience|exp)',
            r'(\d+\+?\s*years?).*?(?:experience|exp)',
            r'(?:experience|exp).*?(\d+\+?\s*years?)',
            r'(fresher|fresh graduate|entry.level|no.experience)',
        ]
        
        for pattern in exp_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Not specified"
    
    def extract_education_requirements(self, text: str) -> List[str]:
        """Extract education requirements"""
        edu_patterns = [
            r'(Bachelor.*?(?:degree|in).*?)(?=\n|$)',
            r'(Master.*?(?:degree|in).*?)(?=\n|$)',
            r'(B\.?Tech.*?)(?=\n|$)',
            r'(M\.?Tech.*?)(?=\n|$)',
            r'(MBA.*?)(?=\n|$)',
            r'(PhD.*?)(?=\n|$)',
            r'Education.*?:?\s*([^\n]+)'
        ]
        
        education = []
        for pattern in edu_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            education.extend([match.strip() for match in matches if len(match.strip()) > 5])
        
        return list(set(education))[:5]  # Limit to 5 entries
    
    def extract_location(self, text: str) -> str:
        """Extract job location"""
        location_patterns = [
            r'Location:?\s*([^\n]+)',
            r'Based in:?\s*([^\n]+)',
            r'Office:?\s*([^\n]+)',
            r'(?:Bangalore|Hyderabad|Delhi|Mumbai|Pune|Chennai|Kolkata|Gurgaon|Noida)',
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                location = match.group(1).strip() if len(match.groups()) > 0 else match.group(0)
                if len(location) < 100:
                    return location
        
        return "Not specified"

@st.cache_resource
def load_sentence_model():
    """Load sentence transformer model with caching"""
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Error loading sentence transformer model: {e}")
        return None

class RelevanceScorer:
    """Calculate relevance scores using hybrid approach"""
    
    def __init__(self):
        self.sentence_model = load_sentence_model()
        self.tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=1000)
    
    def calculate_skill_overlap(self, resume_skills: List[str], required_skills: List[str]) -> Tuple[List[str], List[str]]:
        """Calculate skill overlap between resume and job requirements"""
        resume_skills_lower = [skill.lower().strip() for skill in resume_skills if skill]
        required_skills_lower = [skill.lower().strip() for skill in required_skills if skill]
        
        matches = []
        missing = []
        
        for req_skill in required_skills_lower:
            matched = False
            for res_skill in resume_skills_lower:
                # Exact match or partial match
                if req_skill == res_skill or req_skill in res_skill or res_skill in req_skill:
                    if req_skill not in [m.lower() for m in matches]:
                        matches.append(req_skill.title())
                    matched = True
                    break
            
            if not matched:
                missing.append(req_skill.title())
        
        return matches, missing
    
    def hard_match_score(self, resume_data: Dict, jd_data: Dict) -> Dict:
        """Calculate hard match score based on exact keyword matching"""
        resume_skills = resume_data.get('skills', [])
        required_skills = jd_data.get('required_skills', [])
        preferred_skills = jd_data.get('preferred_skills', [])
        
        # Calculate skill matches
        required_matches, missing_required = self.calculate_skill_overlap(resume_skills, required_skills)
        preferred_matches, missing_preferred = self.calculate_skill_overlap(resume_skills, preferred_skills)
        
        # Calculate scores
        required_score = len(required_matches) / max(len(required_skills), 1)
        preferred_score = len(preferred_matches) / max(len(preferred_skills), 1)
        
        # Overall hard match score (weighted)
        hard_score = (required_score * 0.7 + preferred_score * 0.3) * 100
        
        return {
            'hard_score': min(hard_score, 100),
            'required_matches': required_matches,
            'preferred_matches': preferred_matches,
            'missing_required': missing_required,
            'missing_preferred': missing_preferred,
            'required_score': required_score * 100,
            'preferred_score': preferred_score * 100
        }
    
    def semantic_match_score(self, resume_data: Dict, jd_data: Dict) -> float:
        """Calculate semantic similarity using embeddings"""
        if not self.sentence_model:
            return 0.0
        
        try:
            resume_text = resume_data.get('clean_text', '')
            jd_text = jd_data.get('raw_text', '')
            
            if not resume_text or not jd_text:
                return 0.0
            
            # Create embeddings
            resume_embedding = self.sentence_model.encode([resume_text])
            jd_embedding = self.sentence_model.encode([jd_text])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(resume_embedding, jd_embedding)[0][0]
            
            return max(0, similarity * 100)  # Convert to percentage, ensure non-negative
        except Exception as e:
            logger.error(f"Error in semantic matching: {e}")
            return 0.0
    
    def calculate_relevance_score(self, resume_data: Dict, jd_data: Dict) -> Dict:
        """Calculate overall relevance score"""
        # Get hard match scores
        hard_match = self.hard_match_score(resume_data, jd_data)
        
        # Get semantic match score
        semantic_score = self.semantic_match_score(resume_data, jd_data)
        
        # Combine scores with weights (60% hard match, 40% semantic)
        final_score = (hard_match['hard_score'] * 0.6 + semantic_score * 0.4)
        
        # Determine verdict based on score
        if final_score >= 75:
            verdict = "High"
            verdict_color = "🟢"
        elif final_score >= 50:
            verdict = "Medium"
            verdict_color = "🟡"
        else:
            verdict = "Low"
            verdict_color = "🔴"
        
        return {
            'relevance_score': round(final_score, 2),
            'hard_match_score': round(hard_match['hard_score'], 2),
            'semantic_score': round(semantic_score, 2),
            'verdict': verdict,
            'verdict_color': verdict_color,
            'required_matches': hard_match['required_matches'],
            'preferred_matches': hard_match['preferred_matches'],
            'missing_required': hard_match['missing_required'],
            'missing_preferred': hard_match['missing_preferred'],
            'required_score': round(hard_match['required_score'], 2),
            'preferred_score': round(hard_match['preferred_score'], 2)
        }
    
    def generate_feedback(self, resume_data: Dict, jd_data: Dict, scores: Dict) -> str:
        """Generate personalized feedback for students"""
        feedback_parts = []
        
        # Overall assessment
        score = scores['relevance_score']
        verdict = scores['verdict']
        
        if verdict == "High":
            feedback_parts.append("🎉 **Excellent Match!** Your resume aligns very well with this job requirement.")
        elif verdict == "Medium":
            feedback_parts.append("👍 **Good Match** with room for improvement. You have a solid foundation for this role.")
        else:
            feedback_parts.append("⚠️ **Needs Improvement.** Your resume requires significant enhancements to match this role better.")
        
        # Score breakdown
        feedback_parts.append(f"\n**📊 Score Breakdown:**")
        feedback_parts.append(f"• Overall Relevance: **{score}%**")
        feedback_parts.append(f"• Skills Match: **{scores['hard_match_score']}%**")
        feedback_parts.append(f"• Content Relevance: **{scores['semantic_score']}%**")
        
        # Skills analysis
        if scores['required_matches']:
            feedback_parts.append(f"\n**✅ Matching Required Skills ({len(scores['required_matches'])}):**")
            feedback_parts.append(f"• {', '.join(scores['required_matches'][:10])}")
        
        if scores['missing_required']:
            feedback_parts.append(f"\n**🚨 Critical Skills to Add ({len(scores['missing_required'])}):**")
            feedback_parts.append(f"• {', '.join(scores['missing_required'][:8])}")
        
        if scores['preferred_matches']:
            feedback_parts.append(f"\n**⭐ Bonus Skills You Have ({len(scores['preferred_matches'])}):**")
            feedback_parts.append(f"• {', '.join(scores['preferred_matches'][:8])}")
        
        # Actionable recommendations
        feedback_parts.append(f"\n**💡 Recommendations:**")
        
        if score < 40:
            feedback_parts.append("• **Critical:** Add projects showcasing the missing required skills")
            feedback_parts.append("• **Learn:** Focus on acquiring the top 3-5 missing critical skills")
            feedback_parts.append("• **Highlight:** Better emphasize transferable skills from your experience")
        elif score < 70:
            feedback_parts.append("• **Enhance:** Add specific examples and quantify your achievements")
            feedback_parts.append("• **Skills:** Consider adding 2-3 missing preferred skills")
            feedback_parts.append("• **Keywords:** Include more relevant technical terms throughout your resume")
        else:
            feedback_parts.append("• **Polish:** Fine-tune your resume with specific metrics and achievements")
            feedback_parts.append("• **Standout:** Add unique projects or certifications to differentiate yourself")
            feedback_parts.append("• **Apply:** You're well-qualified for this position!")
        
        # Experience and education feedback
        resume_exp = resume_data.get('experience', 'Not specified')
        jd_exp = jd_data.get('experience', 'Not specified')
        
        if resume_exp != 'Not specified' and jd_exp != 'Not specified':
            feedback_parts.append(f"\n**📈 Experience Match:**")
            feedback_parts.append(f"• Your experience: {resume_exp}")
            feedback_parts.append(f"• Required experience: {jd_exp}")
        
        return '\n'.join(feedback_parts)

class DatabaseManager:
    """Manage database operations with Streamlit session state"""
    # NOTE: Initialization is now handled outside the class
    
    def save_job_description(self, jd_data: Dict, company: str = "", location: str = "") -> int:
        """Save job description to session state"""
        job_id = st.session_state.next_job_id
        
        job_entry = {
            'id': job_id,
            'title': jd_data.get('title', 'Unknown Position'),
            'company': company or 'Not specified',
            'location': location or jd_data.get('location', 'Not specified'),
            'raw_text': jd_data.get('raw_text', ''),
            'required_skills': jd_data.get('required_skills', []),
            'preferred_skills': jd_data.get('preferred_skills', []),
            'experience': jd_data.get('experience', 'Not specified'),
            'education': jd_data.get('education', []),
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        st.session_state.jobs_data.append(job_entry)
        st.session_state.next_job_id += 1
        
        return job_id
    
    def save_evaluation(self, job_id: int, candidate_name: str, resume_data: Dict, scores: Dict, feedback: str):
        """Save resume evaluation results"""
        eval_id = st.session_state.next_eval_id
        
        eval_entry = {
            'id': eval_id,
            'job_id': job_id,
            'candidate_name': candidate_name or 'Unknown Candidate',
            'file_name': resume_data.get('file_name', ''),
            'relevance_score': scores['relevance_score'],
            'hard_match_score': scores['hard_match_score'],
            'semantic_score': scores['semantic_score'],
            'verdict': scores['verdict'],
            'verdict_color': scores['verdict_color'],
            'required_matches': scores['required_matches'],
            'preferred_matches': scores['preferred_matches'],
            'missing_required': scores['missing_required'],
            'missing_preferred': scores['missing_preferred'],
            'required_score': scores['required_score'],
            'preferred_score': scores['preferred_score'],
            'feedback': feedback,
            'resume_skills': resume_data.get('skills', []),
            'resume_experience': resume_data.get('experience', ''),
            'resume_education': resume_data.get('education', []),
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        st.session_state.evaluations_data.append(eval_entry)
        st.session_state.next_eval_id += 1
    
    def get_evaluations_by_job(self, job_id: int) -> List[Dict]:
        """Get all evaluations for a specific job"""
        evaluations = [eval for eval in st.session_state.evaluations_data if eval['job_id'] == job_id]
        return sorted(evaluations, key=lambda x: x['relevance_score'], reverse=True)
    
    def get_all_jobs(self) -> List[Dict]:
        """Get all job descriptions"""
        return sorted(st.session_state.jobs_data, key=lambda x: x['id'], reverse=True)
    
    def get_job_by_id(self, job_id: int) -> Optional[Dict]:
        """Get job by ID"""
        for job in st.session_state.jobs_data:
            if job['id'] == job_id:
                return job
        return None
    
    def get_all_evaluations(self) -> List[Dict]:
        """Get all evaluations"""
        return sorted(st.session_state.evaluations_data, key=lambda x: x['created_at'], reverse=True)

class ResumeRelevanceSystem:
    """Main system orchestrating all components"""
    
    def __init__(self):
        self.resume_parser = ResumeParser()
        self.jd_parser = JobDescriptionParser()
        self.scorer = RelevanceScorer()
        self.db = DatabaseManager()
    
    def process_job_description(self, jd_text: str, company: str = "", location: str = "") -> int:
        """Process and save job description"""
        jd_data = self.jd_parser.parse_job_description(jd_text)
        if not jd_data:
            raise ValueError("Could not parse job description")
        
        job_id = self.db.save_job_description(jd_data, company, location)
        logger.info(f"Job description saved with ID: {job_id}")
        return job_id
    
    def evaluate_resume(self, job_id: int, uploaded_file, candidate_name: str = "") -> Dict:
        """Evaluate a single resume against a job description"""
        # Get job description
        job_data = self.db.get_job_by_id(job_id)
        if not job_data:
            raise ValueError(f"Job description with ID {job_id} not found")
        
        # Parse resume
        resume_data = self.resume_parser.parse_resume(uploaded_file)
        if not resume_data:
            raise ValueError("Could not parse resume")
        
        # Calculate scores
        scores = self.scorer.calculate_relevance_score(resume_data, job_data)
        
        # Generate feedback
        feedback = self.scorer.generate_feedback(resume_data, job_data, scores)
        
        # Save evaluation
        if not candidate_name:
            candidate_name = uploaded_file.name.split('.')[0] if uploaded_file else "Unknown"
        
        self.db.save_evaluation(job_id, candidate_name, resume_data, scores, feedback)
        
        return {
            'candidate_name': candidate_name,
            'resume_data': resume_data,
            'scores': scores,
            'feedback': feedback
        }
    
    def get_job_evaluations(self, job_id: int) -> List[Dict]:
        """Get all evaluations for a job, sorted by relevance score"""
        return self.db.get_evaluations_by_job(job_id)
    
    def get_all_jobs(self) -> List[Dict]:
        """Get all job descriptions"""
        return self.db.get_all_jobs()
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        jobs = self.get_all_jobs()
        all_evaluations = self.db.get_all_evaluations()
        
        if not all_evaluations:
            return {
                'total_jobs': len(jobs),
                'total_evaluations': 0,
                'average_score': 0,
                'high_count': 0,
                'medium_count': 0,
                'low_count': 0
            }
        
        scores = [eval['relevance_score'] for eval in all_evaluations]
        verdicts = [eval['verdict'] for eval in all_evaluations]
        
        return {
            'total_jobs': len(jobs),
            'total_evaluations': len(all_evaluations),
            'average_score': sum(scores) / len(scores),
            'high_count': verdicts.count('High'),
            'medium_count': verdicts.count('Medium'),
            'low_count': verdicts.count('Low'),
            'max_score': max(scores),
            'min_score': min(scores)
        }

# Initialize the system
@st.cache_resource
def get_system():
    """Get system instance with caching"""
    return ResumeRelevanceSystem()

def initialize_session_state():
    """Initialize session state for data storage if not already present."""
    if 'jobs_data' not in st.session_state:
        st.session_state.jobs_data = []
    if 'evaluations_data' not in st.session_state:
        st.session_state.evaluations_data = []
    if 'next_job_id' not in st.session_state:
        st.session_state.next_job_id = 1
    if 'next_eval_id' not in st.session_state:
        st.session_state.next_eval_id = 1

def main():
    """Main Streamlit application"""
    
    initialize_session_state()
    
    # Initialize system
    system = get_system()
    
    # Header
    st.title("🎯 Resume Relevance Check System")
    st.markdown("### *Automated AI-powered resume evaluation for efficient recruitment*")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Dashboard", "📝 Add Job Description", "📊 Evaluate Resumes", "📈 View Results", "📋 Bulk Analysis"],
        index=0
    )
    
    # Page routing
    if page == "🏠 Dashboard":
        show_dashboard(system)
    elif page == "📝 Add Job Description":
        show_add_job_page(system)
    elif page == "📊 Evaluate Resumes":
        show_evaluate_resume_page(system)
    elif page == "📈 View Results":
        show_results_page(system)
    elif page == "📋 Bulk Analysis":
        show_bulk_analysis_page(system)

def show_dashboard(system):
    """Dashboard page with overview and statistics"""
    st.header("🏠 Dashboard Overview")
    
    # Get statistics
    stats = system.get_statistics()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📋 Total Jobs",
            value=stats['total_jobs'],
            help="Number of job descriptions added"
        )
    
    with col2:
        st.metric(
            label="📊 Total Evaluations",
            value=stats['total_evaluations'],
            help="Number of resumes evaluated"
        )
    
    with col3:
        if stats['total_evaluations'] > 0:
            st.metric(
                label="📈 Average Score",
                value=f"{stats['average_score']:.1f}%",
                help="Average relevance score across all evaluations"
            )
        else:
            st.metric(label="📈 Average Score", value="N/A")
    
    with col4:
        if stats['total_evaluations'] > 0:
            high_percentage = (stats['high_count'] / stats['total_evaluations']) * 100
            st.metric(
                label="🎯 High Matches",
                value=f"{stats['high_count']} ({high_percentage:.1f}%)",
                help="Number of high-scoring candidates"
            )
        else:
            st.metric(label="🎯 High Matches", value="0")
    
    st.markdown("---")
    
    # Charts section
    if stats['total_evaluations'] > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Verdict distribution
            st.subheader("📊 Candidate Distribution")
            verdict_data = {
                'Verdict': ['High', 'Medium', 'Low'],
                'Count': [stats['high_count'], stats['medium_count'], stats['low_count']],
                'Color': ['#28a745', '#ffc107', '#dc3545']
            }
            
            fig_pie = px.pie(
                values=verdict_data['Count'],
                names=verdict_data['Verdict'],
                color=verdict_data['Verdict'],
                color_discrete_map={
                    'High': '#28a745',
                    'Medium': '#ffc107',
                    'Low': '#dc3545'
                },
                title="Distribution by Suitability"
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Score distribution
            st.subheader("📈 Score Distribution")
            all_evaluations = system.db.get_all_evaluations()
            scores = [eval['relevance_score'] for eval in all_evaluations]
            
            fig_hist = px.histogram(
                x=scores,
                nbins=20,
                title="Relevance Score Distribution",
                labels={'x': 'Relevance Score (%)', 'y': 'Number of Candidates'}
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
    
    # Recent activity
    st.subheader("📋 Recent Job Descriptions")
    
    jobs = system.get_all_jobs()
    if jobs:
        # Show recent jobs in a nice format
        for i, job in enumerate(jobs[:5]):  # Show last 5 jobs
            with st.expander(f"📄 {job['title']} - {job['company']}", expanded=(i == 0)):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Company:** {job['company']}")
                    st.write(f"**Location:** {job['location']}")
                    st.write(f"**Experience Required:** {job['experience']}")
                    st.write(f"**Added:** {job['created_at'].split()[0]}")
                
                with col2:
                    evaluations = system.get_job_evaluations(job['id'])
                    st.metric("Evaluations", len(evaluations))
                    
                    if evaluations:
                        avg_score = sum(e['relevance_score'] for e in evaluations) / len(evaluations)
                        st.metric("Avg Score", f"{avg_score:.1f}%")
                    
                    if st.button(f"View Details →", key=f"job_detail_{job['id']}"):
                        st.session_state.selected_job_id = job['id']
                        st.session_state.page = "📈 View Results"
                        st.rerun()
                
                # Skills preview
                
                if job['required_skills']:
                    st.write("**Key Required Skills:**")
                    skill_tags = "".join([f'<span style=" padding: 2px 6px; margin: 2px; border-radius: 3px; font-size: 0.8em;">{skill}</span>' for skill in job['required_skills'][:8]])
                    st.markdown(skill_tags, unsafe_allow_html=True)
    else:
        st.info("🔍 No job descriptions added yet. Start by adding your first job description!")
        if st.button("➕ Add Job Description"):
            st.session_state.page = "📝 Add Job Description"
            st.rerun()

def show_add_job_page(system):
    """Add job description page"""
    st.header("📝 Add New Job Description")
    st.markdown("Add a job description to start evaluating resumes against it.")
    
    with st.form("job_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            company = st.text_input(
                "🏢 Company Name",
                placeholder="e.g., Innomatics Research Labs",
                help="Enter the hiring company name"
            )
        
        with col2:
            location = st.text_input(
                "📍 Location",
                placeholder="e.g., Hyderabad, Bangalore, Delhi NCR",
                help="Job location or 'Remote' if applicable"
            )
        
        job_description = st.text_area(
            "📄 Job Description",
            height=400,
            placeholder="""Paste the complete job description here...

Example:
Software Developer - Python

We are looking for a skilled Python developer to join our team.

Required Skills:
- Python programming
- Django or Flask framework
- SQL databases
- Git version control

Preferred Skills:
- Machine Learning
- Docker
- AWS cloud services

Experience: 2-4 years
Education: Bachelor's degree in Computer Science""",
            help="Include all details: role, responsibilities, required skills, preferred skills, experience, education"
        )
        
        submitted = st.form_submit_button("➕ Add Job Description", type="primary")
        
        if submitted:
            if not job_description.strip():
                st.error("❌ Job description is required!")
            else:
                try:
                    with st.spinner("🔄 Processing job description..."):
                        job_id = system.process_job_description(job_description, company, location)
                    
                    st.success(f"✅ Job description added successfully! (ID: {job_id})")
                    
                    # Show parsed information
                    job_data = system.db.get_job_by_id(job_id)
                    if job_data:
                        st.subheader("📋 Parsed Information")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Title:** {job_data['title']}")
                            st.write(f"**Experience:** {job_data['experience']}")
                        
                        with col2:
                            st.write(f"**Company:** {job_data['company']}")
                            st.write(f"**Location:** {job_data['location']}")
                        
                        if job_data['required_skills']:
                            st.write("**Required Skills:**")
                            st.write(", ".join(job_data['required_skills']))
                        
                        if job_data['preferred_skills']:
                            st.write("**Preferred Skills:**")
                            st.write(", ".join(job_data['preferred_skills']))
                    
                    st.info("👉 Now you can evaluate resumes against this job description!")
                    
                except Exception as e:
                    st.error(f"❌ Error adding job description: {str(e)}")

def show_evaluate_resume_page(system):
    """Resume evaluation page"""
    st.header("📊 Evaluate Resume")
    st.markdown("Upload a resume and select a job to get an AI-powered relevance analysis.")
    
    jobs = system.get_all_jobs()
    
    if not jobs:
        st.warning("⚠️ No job descriptions available. Please add a job description first.")
        if st.button("➕ Add Job Description"):
            st.session_state.page = "📝 Add Job Description"
            st.rerun()
        return
    
    # Job selection
    st.subheader("1️⃣ Select Job Position")
    job_options = {f"{job['title']} - {job['company']} ({job['location']})": job['id'] for job in jobs}
    selected_job_key = st.selectbox(
        "Choose the job position to evaluate against:",
        list(job_options.keys()),
        help="Select the job description you want to match the resume against"
    )
    
    if selected_job_key:
        job_id = job_options[selected_job_key]
        selected_job = system.db.get_job_by_id(job_id)
        
        # Show job details
        with st.expander("📋 View Job Details", expanded=False):
            if selected_job:
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Title:** {selected_job['title']}")
                    st.write(f"**Experience:** {selected_job['experience']}")
                with col2:
                    st.write(f"**Company:** {selected_job['company']}")
                    st.write(f"**Location:** {selected_job['location']}")
                
                if selected_job['required_skills']:
                    st.write("**Required Skills:**")
                    skill_tags = "".join([f'<span style="background-color: #ffebee; color: #c62828; padding: 4px 8px; margin: 2px; border-radius: 4px; font-size: 0.9em; border: 1px solid #ffcdd2;">{skill}</span>' for skill in selected_job['required_skills']])
                    st.markdown(skill_tags, unsafe_allow_html=True)
                
                if selected_job['preferred_skills']:
                    st.write("**Preferred Skills:**")
                    skill_tags = "".join([f'<span style="background-color: #e8f5e8; color: #2e7d32; padding: 4px 8px; margin: 2px; border-radius: 4px; font-size: 0.9em; border: 1px solid #c8e6c9;">{skill}</span>' for skill in selected_job['preferred_skills']])
                    st.markdown(skill_tags, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Resume upload section
        st.subheader("2️⃣ Upload Resume")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose resume file",
                type=['pdf', 'docx'],
                help="Upload PDF or DOCX resume files (max 10MB)"
            )
        
        with col2:
            candidate_name = st.text_input(
                "👤 Candidate Name (Optional)",
                placeholder="John Doe",
                help="Enter candidate name for better tracking"
            )
        
        # Evaluation section
        if uploaded_file:
            st.subheader("3️⃣ Resume Analysis")
            
            # Show file info
            st.info(f"📄 **File:** {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
            
            if st.button("🚀 Analyze Resume", type="primary"):
                try:
                    with st.spinner("🔄 Analyzing resume... This may take a few seconds."):
                        result = system.evaluate_resume(job_id, uploaded_file, candidate_name)
                    
                    # Display results
                    st.success("✅ Analysis completed!")
                    
                    scores = result['scores']
                    
                    # Score display with visual indicators
                    st.subheader("📊 Relevance Analysis Results")
                    
                    # Main score with color coding
                    score_color = "#28a745" if scores['verdict'] == "High" else "#ffc107" if scores['verdict'] == "Medium" else "#dc3545"
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; background-color: {score_color}20; border-radius: 10px; border: 2px solid {score_color};">
                        <h2 style="color: {score_color}; margin: 0;">{scores['verdict_color']} {scores['relevance_score']}% - {scores['verdict']} Match</h2>
                        <p style="margin: 5px 0 0 0; color: #666;">Overall Relevance Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Detailed breakdown
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="🎯 Skills Match",
                            value=f"{scores['hard_match_score']}%",
                            help="How well your skills match job requirements"
                        )
                    
                    with col2:
                        st.metric(
                            label="📝 Content Relevance",
                            value=f"{scores['semantic_score']}%",
                            help="How relevant your overall experience is"
                        )
                    
                    with col3:
                        st.metric(
                            label="✅ Required Skills",
                            value=f"{scores['required_score']}%",
                            help="Percentage of required skills you have"
                        )
                    
                    # Skills analysis
                    st.subheader("🛠️ Skills Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if scores['required_matches']:
                            st.markdown("**✅ Required Skills You Have:**")
                            for skill in scores['required_matches']:
                                st.markdown(f"• ✅ {skill}")
                        
                        if scores['preferred_matches']:
                            st.markdown("**⭐ Bonus Skills You Have:**")
                            for skill in scores['preferred_matches']:
                                st.markdown(f"• ⭐ {skill}")
                    
                    with col2:
                        if scores['missing_required']:
                            st.markdown("**🚨 Missing Critical Skills:**")
                            for skill in scores['missing_required']:
                                st.markdown(f"• ❌ {skill}")
                        
                        if scores['missing_preferred']:
                            st.markdown("**💡 Additional Skills to Consider:**")
                            for skill in scores['missing_preferred'][:5]:  # Limit to 5
                                st.markdown(f"• 💡 {skill}")
                    
                    # Feedback section
                    st.subheader("💬 Personalized Feedback")
                    st.info(result['feedback'])
                    
                    # Resume insights
                    if result['resume_data']:
                        with st.expander("👀 Resume Insights", expanded=False):
                            resume = result['resume_data']
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**📚 Extracted Skills:**")
                                if resume['skills']:
                                    st.write(", ".join(resume['skills'][:15]))
                                else:
                                    st.write("No skills detected")
                                
                                st.write("**🎓 Education:**")
                                if resume['education']:
                                    for edu in resume['education']:
                                        st.write(f"• {edu}")
                                else:
                                    st.write("No education details detected")
                            
                            with col2:
                                st.write("**💼 Experience:**")
                                st.write(resume['experience'])
                                
                                st.write("**📄 File Info:**")
                                st.write(f"• File: {resume['file_name']}")
                                st.write(f"• Text Length: {len(resume['clean_text'])} characters")
                    
                    # Action buttons
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("📊 View All Results for This Job"):
                            st.session_state.selected_job_id = job_id
                            st.session_state.page = "📈 View Results"
                            st.rerun()
                    
                    with col2:
                        if st.button("📝 Evaluate Another Resume"):
                            st.rerun()
                    
                    with col3:
                        if st.button("🏠 Back to Dashboard"):
                            st.session_state.page = "🏠 Dashboard"
                            st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Error evaluating resume: {str(e)}")
                    st.info("💡 Make sure the file is a valid PDF or DOCX and contains readable text.")

def show_results_page(system):
    """Results viewing page"""
    st.header("📈 Evaluation Results")
    st.markdown("View and analyze resume evaluation results by job position.")
    
    jobs = system.get_all_jobs()
    
    if not jobs:
        st.info("📋 No job descriptions available.")
        return
    
    # Job selection
    job_options = {f"{job['title']} - {job['company']} ({job['location']})": job['id'] for job in jobs}
    
    # Check if there's a selected job from session state
    selected_key = None
    if 'selected_job_id' in st.session_state:
        for key, job_id_val in job_options.items():
            if job_id_val == st.session_state.selected_job_id:
                selected_key = key
                break
    
    selected_job_key = st.selectbox(
        "📋 Select Job to View Results:",
        list(job_options.keys()),
        index=list(job_options.keys()).index(selected_key) if selected_key else 0,
        help="Choose a job to view all resume evaluations for that position"
    )
    
    if selected_job_key:
        job_id = job_options[selected_job_key]
        evaluations = system.get_job_evaluations(job_id)
        selected_job = system.db.get_job_by_id(job_id)
        
        # Job summary
        if selected_job:
            st.subheader(f"📋 {selected_job['title']}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.info(f"**Company:** {selected_job['company']}")
            with col2:
                st.info(f"**Location:** {selected_job['location']}")
            with col3:
                st.info(f"**Experience:** {selected_job['experience']}")
            with col4:
                st.info(f"**Total Candidates:** {len(evaluations)}")
        
        st.markdown("---")
        
        if evaluations:
            # Statistics
            st.subheader("📊 Evaluation Statistics")
            
            scores = [e['relevance_score'] for e in evaluations]
            high_count = sum(1 for e in evaluations if e['verdict'] == 'High')
            medium_count = sum(1 for e in evaluations if e['verdict'] == 'Medium')
            low_count = sum(1 for e in evaluations if e['verdict'] == 'Low')
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🟢 High Suitability", high_count, f"{(high_count/len(evaluations)*100):.1f}%")
            with col2:
                st.metric("🟡 Medium Suitability", medium_count, f"{(medium_count/len(evaluations)*100):.1f}%")
            with col3:
                st.metric("🔴 Low Suitability", low_count, f"{(low_count/len(evaluations)*100):.1f}%")
            with col4:
                st.metric("📊 Average Score", f"{sum(scores)/len(scores):.1f}%")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Score distribution
                fig_hist = px.histogram(
                    x=scores,
                    nbins=min(15, len(scores)),
                    title="Score Distribution",
                    labels={'x': 'Relevance Score (%)', 'y': 'Number of Candidates'},
                    color_discrete_sequence=['#007bff']
                )
                fig_hist.update_layout(height=300)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Verdict pie chart
                verdict_counts = [high_count, medium_count, low_count]
                fig_pie = px.pie(
                    values=verdict_counts,
                    names=['High', 'Medium', 'Low'],
                    title="Suitability Distribution",
                    color_discrete_map={
                        'High': '#28a745',
                        'Medium': '#ffc107',
                        'Low': '#dc3545'
                    }
                )
                fig_pie.update_layout(height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            st.markdown("---")
            
            # Filter and sort options
            st.subheader("🔍 Filter & Sort Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                verdict_filter = st.selectbox(
                    "Filter by Verdict:",
                    ["All", "High", "Medium", "Low"],
                    help="Filter candidates by their suitability verdict"
                )
            
            with col2:
                min_score = st.slider(
                    "Minimum Score:",
                    0, 100, 0,
                    help="Show only candidates with score above this threshold"
                )
            
            with col3:
                sort_by = st.selectbox(
                    "Sort by:",
                    ["Relevance Score", "Date", "Name"],
                    help="Choose how to sort the results"
                )
            
            # Apply filters
            filtered_evaluations = evaluations
            
            if verdict_filter != "All":
                filtered_evaluations = [e for e in filtered_evaluations if e['verdict'] == verdict_filter]
            
            filtered_evaluations = [e for e in filtered_evaluations if e['relevance_score'] >= min_score]
            
            # Sort results
            if sort_by == "Relevance Score":
                filtered_evaluations = sorted(filtered_evaluations, key=lambda x: x['relevance_score'], reverse=True)
            elif sort_by == "Date":
                filtered_evaluations = sorted(filtered_evaluations, key=lambda x: x['created_at'], reverse=True)
            elif sort_by == "Name":
                filtered_evaluations = sorted(filtered_evaluations, key=lambda x: x['candidate_name'])
            
            st.markdown("---")
            
            # Results display
            st.subheader(f"📋 Candidates ({len(filtered_evaluations)} results)")
            
            if filtered_evaluations:
                for i, eval_item in enumerate(filtered_evaluations):
                    # Color coding based on verdict
                    if eval_item['verdict'] == 'High':
                        border_color = "#28a745"
                    elif eval_item['verdict'] == 'Medium':
                        border_color = "#ffc107"
                    else:
                        border_color = "#dc3545"
                    
                    with st.expander(
                        f"{eval_item['verdict_color']} **{eval_item['candidate_name']}** - {eval_item['relevance_score']}% ({eval_item['verdict']})",
                        expanded=(i == 0)  # Expand first result
                    ):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Basic info
                            st.write(f"**📄 File:** {eval_item['file_name']}")
                            st.write(f"**📅 Evaluated:** {eval_item['created_at'].split()[0]}")
                            st.write(f"**💼 Experience:** {eval_item['resume_experience']}")
                            
                            # Score breakdown
                            st.write("**📊 Score Breakdown:**")
                            st.write(f"• Overall: **{eval_item['relevance_score']}%**")
                            st.write(f"• Skills Match: **{eval_item['hard_match_score']}%**")
                            st.write(f"• Content Relevance: **{eval_item['semantic_score']}%**")
                        
                        with col2:
                            # Circular progress indicator
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = eval_item['relevance_score'],
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Relevance"},
                                gauge = {
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': border_color},
                                    'steps': [
                                        {'range': [0, 50], 'color': "#f8d7da"},
                                        {'range': [50, 75], 'color': "#fff3cd"},
                                        {'range': [75, 100], 'color': "#d4edda"}
                                    ]
                                }
                            ))
                            fig.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No candidates match the current filter criteria.")
        else:
            st.info("No resumes have been evaluated for this job position yet.")
            if st.button("📊 Evaluate a Resume Now"):
                st.session_state.selected_job_id = job_id
                st.session_state.page = "📊 Evaluate Resumes"
                st.rerun()

def show_bulk_analysis_page(system):
    """Bulk resume analysis page"""
    st.header("📋 Bulk Resume Analysis")
    st.markdown("Upload multiple resumes at once to evaluate them against a selected job description.")
    
    jobs = system.get_all_jobs()
    
    if not jobs:
        st.warning("⚠️ No job descriptions available. Please add a job description first.")
        if st.button("➕ Add Job Description"):
            st.session_state.page = "📝 Add Job Description"
            st.rerun()
        return
    
    # Job selection
    job_options = {f"{job['title']} - {job['company']} ({job['location']})": job['id'] for job in jobs}
    selected_job_key = st.selectbox(
        "1️⃣ Select the job position to evaluate against:",
        list(job_options.keys()),
        help="All uploaded resumes will be matched against this job description"
    )
    
    if selected_job_key:
        job_id = job_options[selected_job_key]
        
        # File uploader for multiple files
        st.subheader("2️⃣ Upload Resumes")
        uploaded_files = st.file_uploader(
            "Choose resume files (PDF or DOCX)",
            type=['pdf', 'docx'],
            accept_multiple_files=True,
            help="You can drag and drop multiple files here."
        )
        
        if uploaded_files:
            st.info(f"📁 You have selected {len(uploaded_files)} files.")
            
            if st.button(f"🚀 Analyze All {len(uploaded_files)} Resumes", type="primary"):
                st.subheader("3️⃣ Analysis Progress & Results")
                progress_bar = st.progress(0, text="Starting bulk analysis...")
                results = []
                errors = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Update progress bar
                        progress_text = f"Analyzing {uploaded_file.name} ({i+1}/{len(uploaded_files)})..."
                        progress_bar.progress((i + 1) / len(uploaded_files), text=progress_text)
                        
                        # Evaluate the resume
                        result = system.evaluate_resume(job_id, uploaded_file)
                        results.append(result)
                        
                    except Exception as e:
                        errors.append((uploaded_file.name, str(e)))
                        logger.error(f"Error processing {uploaded_file.name}: {e}")
                
                progress_bar.empty()  # Remove the progress bar after completion
                
                # Display summary
                st.success(f"✅ Bulk analysis completed! Processed {len(results)} out of {len(uploaded_files)} resumes.")
                
                if errors:
                    st.error(f"⚠️ Encountered {len(errors)} errors during processing:")
                    with st.expander("View Error Details"):
                        for file_name, error_msg in errors:
                            st.write(f"• **{file_name}:** {error_msg}")
                
                st.markdown("---")
                
                # Show a brief summary of top candidates
                st.subheader("🏆 Top Candidates")
                top_candidates = sorted(results, key=lambda x: x['scores']['relevance_score'], reverse=True)
                
                for candidate in top_candidates[:5]:  # Show top 5
                    score = candidate['scores']['relevance_score']
                    verdict = candidate['scores']['verdict']
                    verdict_color = candidate['scores']['verdict_color']
                    st.markdown(f"• {verdict_color} **{candidate['candidate_name']}**: {score}% ({verdict})")
                
                # Button to view full results
                st.info("Click below to see the detailed breakdown for all candidates.")
                if st.button("📈 View Full Results"):
                    st.session_state.selected_job_id = job_id
                    st.session_state.page = "📈 View Results"
                    st.rerun()

if __name__ == "__main__":
    main()

