import os, re, json, sqlite3, hashlib
from datetime import datetime
from typing import List, Tuple
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Optional NLP
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ SentenceTransformer libraries not available: {e}")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Document parsing
try:
    import pdfplumber
    import docx2txt
    DOC_LIBS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Document parsing libraries not available: {e}")
    print("📝 Text extraction from PDF/DOCX files will be limited")
    DOC_LIBS_AVAILABLE = False

# ------------------- DATA MODELS -------------------
@dataclass
class JobRequirement:
    job_id: str
    title: str
    company: str
    description: str
    must_have_skills: List[str]
    good_to_have_skills: List[str]
    education: List[str]
    created_at: datetime

@dataclass
class ResumeData:
    resume_id: str
    student_name: str
    email: str
    phone: str
    skills: List[str]
    education: List[str]
    raw_text: str
    uploaded_at: datetime

@dataclass
class EvaluationResult:
    evaluation_id: str
    resume_id: str
    job_id: str
    relevance_score: float
    hard_match_score: float
    semantic_score: float
    verdict: str
    missing_skills: List[str]
    missing_qualifications: List[str]
    suggestions: List[str]
    evaluated_at: datetime

# ------------------- DATABASE -------------------
class DatabaseManager:
    def __init__(self, db_path="resume_system.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                title TEXT,
                company TEXT,
                description TEXT,
                must_have_skills TEXT,
                good_to_have_skills TEXT,
                education TEXT,
                created_at TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS resumes (
                resume_id TEXT PRIMARY KEY,
                student_name TEXT,
                email TEXT,
                phone TEXT,
                skills TEXT,
                education TEXT,
                raw_text TEXT,
                uploaded_at TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                evaluation_id TEXT PRIMARY KEY,
                resume_id TEXT,
                job_id TEXT,
                relevance_score REAL,
                hard_match_score REAL,
                semantic_score REAL,
                verdict TEXT,
                missing_skills TEXT,
                missing_qualifications TEXT,
                suggestions TEXT,
                evaluated_at TEXT,
                FOREIGN KEY(resume_id) REFERENCES resumes(resume_id),
                FOREIGN KEY(job_id) REFERENCES jobs(job_id)
            )
        """)
        conn.commit()
        conn.close()
    
    def save_job(self, job: JobRequirement):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job.job_id, job.title, job.company, job.description,
            json.dumps(job.must_have_skills),
            json.dumps(job.good_to_have_skills),
            json.dumps(job.education),
            job.created_at.isoformat()
        ))
        conn.commit(); conn.close()
    
    def save_resume(self, resume: ResumeData):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO resumes VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            resume.resume_id, resume.student_name, resume.email,
            resume.phone, json.dumps(resume.skills),
            json.dumps(resume.education), resume.raw_text,
            resume.uploaded_at.isoformat()
        ))
        conn.commit(); conn.close()
    
    def save_evaluation(self, eval: EvaluationResult):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO evaluations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            eval.evaluation_id, eval.resume_id, eval.job_id,
            eval.relevance_score, eval.hard_match_score,
            eval.semantic_score, eval.verdict,
            json.dumps(eval.missing_skills),
            json.dumps(eval.missing_qualifications),
            json.dumps(eval.suggestions),
            eval.evaluated_at.isoformat()
        ))
        conn.commit(); conn.close()
    
    def get_all_jobs(self) -> List[JobRequirement]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM jobs ORDER BY created_at DESC")
        rows = c.fetchall()
        conn.close()
        jobs = []
        for row in rows:
            def safe_json_load(x):
                try: return json.loads(x)
                except: return []
            try: created_at = datetime.fromisoformat(row[7])
            except: created_at = datetime.now()
            jobs.append(JobRequirement(
                job_id=row[0], title=row[1], company=row[2],
                description=row[3], must_have_skills=safe_json_load(row[4]),
                good_to_have_skills=safe_json_load(row[5]),
                education=safe_json_load(row[6]),
                created_at=created_at
            ))
        return jobs
    
    def get_job_by_id(self, job_id: str) -> JobRequirement:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        row = c.fetchone()
        conn.close()
        
        if row:
            def safe_json_load(x):
                try: return json.loads(x)
                except: return []
            try: created_at = datetime.fromisoformat(row[7])
            except: created_at = datetime.now()
            return JobRequirement(
                job_id=row[0], title=row[1], company=row[2],
                description=row[3], must_have_skills=safe_json_load(row[4]),
                good_to_have_skills=safe_json_load(row[5]),
                education=safe_json_load(row[6]),
                created_at=created_at
            )
        return None
    
    def get_evaluations_by_job(self, job_id: str) -> List[dict]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            SELECT e.*, r.student_name, r.email FROM evaluations e
            JOIN resumes r ON e.resume_id = r.resume_id
            WHERE e.job_id = ? ORDER BY e.relevance_score DESC
        """, (job_id,))
        rows = c.fetchall()
        conn.close()
        evals = []
        for r in rows:
            def safe_json_load(x):
                try: return json.loads(x)
                except: return []
            evals.append({
                "evaluation_id": r[0],
                "resume_id": r[1],
                "job_id": r[2],
                "relevance_score": r[3],
                "hard_match_score": r[4],
                "semantic_score": r[5],
                "verdict": r[6],
                "missing_skills": safe_json_load(r[7]),
                "missing_qualifications": safe_json_load(r[8]),
                "suggestions": safe_json_load(r[9]),
                "evaluated_at": r[10],
                "student_name": r[11],
                "email": r[12]
            })
        return evals

    def get_evaluation_count_by_job(self, job_id: str) -> int:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM evaluations WHERE job_id = ?", (job_id,))
        count = c.fetchone()[0]
        conn.close()
        return count

# ------------------- DOCUMENT PARSER -------------------
class DocumentParser:
    def extract_text_from_pdf(self, content: bytes) -> str:
        if not DOC_LIBS_AVAILABLE:
            return ""
        try:
            import io
            text = ""
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for p in pdf.pages: 
                    page_text = p.extract_text() or ""
                    text += page_text + "\n"
            return text
        except Exception as e:
            print(f"⚠️ PDF extraction failed: {e}")
            return ""
    
    def extract_text_from_docx(self, content: bytes) -> str:
        if not DOC_LIBS_AVAILABLE:
            return ""
        try:
            import io
            return docx2txt.process(io.BytesIO(content))
        except Exception as e:
            print(f"⚠️ DOCX extraction failed: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()
    
    def extract_email(self, text: str) -> str:
        m = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text)
        return m[0] if m else ""
    
    def extract_phone(self, text: str) -> str:
        m = re.findall(r'\+?\d[\d\s\-]{7,}\d', text)
        return m[0] if m else ""
    
    def extract_skills(self, text: str) -> List[str]:
        keywords = ["python","java","javascript","c++","aws","docker","sql","react","node.js","html","css","mongodb","postgresql","git","linux","kubernetes","tensorflow","pytorch","machine learning","data science","api","rest","graphql","microservices"]
        found_skills = []
        for kw in keywords:
            if kw.lower() in text.lower():
                found_skills.append(kw.title())
        return found_skills
    
    def extract_education(self, text: str) -> List[str]:
        keywords = ["bachelor","master","phd","btech","mtech","bsc","msc","diploma"]
        found_education = []
        for kw in keywords:
            if kw.lower() in text.lower():
                found_education.append(kw.title())
        return found_education
    
    def parse_resume(self, content: bytes, file_type: str, filename: str="") -> ResumeData:
        raw_text = ""
        
        if not content:
            st.warning("⚠️ Empty file uploaded")
            return None
            
        try:
            if file_type.lower() == "pdf": 
                raw_text = self.extract_text_from_pdf(content)
            elif file_type.lower() in ["docx","doc"]: 
                raw_text = self.extract_text_from_docx(content)
            else:
                st.error(f"❌ Unsupported file type: {file_type}")
                return None
                
            if not raw_text.strip():
                st.warning("⚠️ No text could be extracted from the file. Please ensure the file contains readable text.")
                # Create a basic resume with just filename
                raw_text = f"Resume file: {filename}"
                
            raw_text = self.clean_text(raw_text)
            resume_id = hashlib.md5(f"{raw_text}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
            student_name = Path(filename).stem if filename else "Unknown"
            
            return ResumeData(
                resume_id=resume_id,
                student_name=student_name,
                email=self.extract_email(raw_text),
                phone=self.extract_phone(raw_text),
                skills=self.extract_skills(raw_text),
                education=self.extract_education(raw_text),
                raw_text=raw_text,
                uploaded_at=datetime.now()
            )
        except Exception as e:
            st.error(f"❌ Error parsing resume: {str(e)}")
            return None

# ------------------- RESUME EVALUATOR -------------------
class ResumeEvaluator:
    def __init__(self):
        self.model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                import torch
                # Force CPU usage to avoid CUDA issues in cloud environments
                device = 'cpu'
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                print(f"✅ SentenceTransformer loaded successfully on {device}")
            except Exception as e:
                print(f"⚠️ Failed to load SentenceTransformer: {e}")
                print("🔄 Falling back to keyword-based similarity")
                self.model = None
    
    def calc_hard_score(self, resume: ResumeData, job: JobRequirement) -> Tuple[float,List[str]]:
        total=0; max_score=0; missing=[]
        resume_skills=[s.lower() for s in resume.skills]
        
        # Must-have skills (higher weight)
        for s in job.must_have_skills:
            max_score += 0.7
            if s.lower() in resume_skills: 
                total+=0.7
            else: 
                missing.append(s)
        
        # Good-to-have skills (lower weight)
        for s in job.good_to_have_skills:
            max_score +=0.3
            if s.lower() in resume_skills: 
                total+=0.3
        
        # Education matching
        max_score +=0.3
        resume_education_text = ' '.join(resume.education).lower()
        if any(e.lower() in resume_education_text for e in job.education):
            total+=0.3
            
        return round((total/max_score*100 if max_score else 0),2), missing
    
    def calc_semantic_score(self, resume: ResumeData, job: JobRequirement) -> float:
        if self.model:
            try:
                r_text = f"{' '.join(resume.skills)} {' '.join(resume.education)} {resume.raw_text[:2000]}"
                j_text = f"{job.title} {' '.join(job.must_have_skills)} {' '.join(job.good_to_have_skills)} {job.description[:2000]}"
                r_emb = self.model.encode([r_text])
                j_emb = self.model.encode([j_text])
                similarity = cosine_similarity(r_emb, j_emb)[0][0]
                return round(float(similarity * 100), 2)
            except Exception as e:
                print(f"⚠️ Semantic scoring failed: {e}")
                return self._fallback_semantic_score(resume, job)
        else:
            return self._fallback_semantic_score(resume, job)
    
    def _fallback_semantic_score(self, resume: ResumeData, job: JobRequirement) -> float:
        """Fallback method using keyword matching when SentenceTransformer is not available"""
        resume_text = f"{' '.join(resume.skills)} {' '.join(resume.education)} {resume.raw_text}".lower()
        job_text = f"{job.title} {' '.join(job.must_have_skills)} {' '.join(job.good_to_have_skills)} {job.description}".lower()
        
        # Extract keywords from job text
        job_words = set(job_text.split())
        # Remove common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall'}
        job_keywords = job_words - stop_words
        
        # Count matches in resume
        matches = 0
        total_keywords = len(job_keywords)
        
        if total_keywords > 0:
            for keyword in job_keywords:
                if keyword in resume_text:
                    matches += 1
            
            # Calculate similarity as percentage of matched keywords
            similarity = (matches / total_keywords) * 100
            return round(min(similarity, 85.0), 2)  # Cap at 85% for keyword-based matching
        
        return 30.0  # Default score if no keywords found
    
    def generate_verdict(self,score:float)->str:
        if score>=75:return "High"
        elif score>=50:return "Medium"
        else: return "Low"
    
    def generate_suggestions(self,missing_skills:List[str],resume:ResumeData)->List[str]:
        sug=[]
        if missing_skills: 
            sug.append(f"Learn or highlight these missing skills: {', '.join(missing_skills[:3])}")
        if len(resume.skills) < 3: 
            sug.append("Add more relevant technical skills to your resume.")
        if not resume.email:
            sug.append("Make sure your email is clearly visible on your resume.")
        sug.append("Consider adding 1-2 relevant projects showing hands-on experience.")
        return sug
    
    def evaluate(self,resume:ResumeData,job:JobRequirement)->EvaluationResult:
        hard_score, missing_skills=self.calc_hard_score(resume,job)
        semantic_score=self.calc_semantic_score(resume,job)
        relevance_score=round(hard_score*0.6+semantic_score*0.4,2)
        verdict=self.generate_verdict(relevance_score)
        missing_qual=[]
        for edu in job.education:
            if edu.lower() not in ' '.join(resume.education).lower(): 
                missing_qual.append(edu)
        suggestions=self.generate_suggestions(missing_skills,resume)
        eval_id=hashlib.md5(f"{resume.resume_id}_{job.job_id}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        return EvaluationResult(
            evaluation_id=eval_id,
            resume_id=resume.resume_id,
            job_id=job.job_id,
            relevance_score=relevance_score,
            hard_match_score=hard_score,
            semantic_score=semantic_score,
            verdict=verdict,
            missing_skills=missing_skills,
            missing_qualifications=missing_qual,
            suggestions=suggestions,
            evaluated_at=datetime.now()
        )

# ------------------- STREAMLIT APP -------------------
def main():
    st.set_page_config("Automated Resume Evaluation", layout="wide")
    st.title("Automated Resume Relevance Check System")
    
    # Show system status
    with st.sidebar:
        st.markdown("###  System Status")
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            st.success(" AI-powered semantic analysis")
        else:
            st.warning(" Using keyword-based analysis")
        
        if DOC_LIBS_AVAILABLE:
            st.success(" PDF/DOCX parsing enabled")
        else:
            st.error(" Document parsing limited")
    
    try:
        db = DatabaseManager()
        parser = DocumentParser()
        evaluator = ResumeEvaluator()
    except Exception as e:
        st.error(f" System initialization failed: {e}")
        st.info("Please refresh the page or contact support if the issue persists.")
        return

    tab = st.sidebar.radio("Select Mode", ["User", "Admin"])
    
    if tab=="Admin":
        st.header("🔹 Admin Panel")
        
        # Add Job Section
        with st.expander("Add New Job Description", expanded=False):
            with st.form("job_form"):
                col1, col2 = st.columns(2)
                with col1:
                    title = st.text_input("Job Title *", placeholder="e.g., Software Engineer")
                    company = st.text_input("Company *", placeholder="e.g., Tech Corp")
                with col2:
                    must_skills = st.text_input("Must-have Skills (comma-separated) *", 
                                              placeholder="e.g., Python, SQL, AWS")
                    good_skills = st.text_input("Good-to-have Skills (comma-separated)", 
                                              placeholder="e.g., Docker, Kubernetes")
                
                desc = st.text_area("Job Description *", height=100,
                                  placeholder="Enter detailed job description...")
                education = st.text_input("Required Education (comma-separated)", 
                                        placeholder="e.g., Bachelor, Master")
                
                submitted = st.form_submit_button("Save Job", type="primary")
                
                if submitted:
                    if title and company and desc and must_skills:
                        job = JobRequirement(
                            job_id=hashlib.md5(f"{title}_{company}_{desc}".encode()).hexdigest()[:12],
                            title=title.strip(), 
                            company=company.strip(), 
                            description=desc.strip(),
                            must_have_skills=[s.strip() for s in must_skills.split(",") if s.strip()],
                            good_to_have_skills=[s.strip() for s in good_skills.split(",") if s.strip()],
                            education=[e.strip() for e in education.split(",") if e.strip()],
                            created_at=datetime.now()
                        )
                        db.save_job(job)
                        st.success(f"Job '{title}' at '{company}' saved successfully!")
                        st.experimental_rerun()
                    else:
                        st.error("Please fill in all required fields (marked with *)")

        st.markdown("---")
        
        # Job Selection and Evaluation Display
        st.subheader("View Job Applications & Evaluations")
        
        jobs = db.get_all_jobs()
        if not jobs:
            st.info("No jobs available. Please add a job first.")
            return
        
        # Create job options with application count
        job_options = ["-- Select a Job --"]
        job_map = {}
        
        for job in jobs:
            eval_count = db.get_evaluation_count_by_job(job.job_id)
            option_text = f"{job.title} @ {job.company} ({eval_count} applications)"
            job_options.append(option_text)
            job_map[option_text] = job
        
        selected_job_option = st.selectbox("Select Job to View Applications:", job_options)
        
        if selected_job_option != "-- Select a Job --":
            selected_job = job_map[selected_job_option]
            
            # Display job details
            with st.expander(f"Job Details: {selected_job.title}", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Company:** {selected_job.company}")
                    st.write(f"**Must-have Skills:** {', '.join(selected_job.must_have_skills)}")
                    st.write(f"**Good-to-have Skills:** {', '.join(selected_job.good_to_have_skills) if selected_job.good_to_have_skills else 'None'}")
                with col2:
                    st.write(f"**Education:** {', '.join(selected_job.education) if selected_job.education else 'Not specified'}")
                    st.write(f"**Posted:** {selected_job.created_at.strftime('%Y-%m-%d %H:%M')}")
                
                st.write(f"**Description:** {selected_job.description}")
            
            # Get evaluations for selected job
            evaluations = db.get_evaluations_by_job(selected_job.job_id)
            
            if evaluations:
                st.success(f"Found {len(evaluations)} applications for this job")
                
                # Convert to DataFrame for easier handling
                df = pd.DataFrame(evaluations)
                
                # Display summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Applications", len(evaluations))
                with col2:
                    high_relevance = len(df[df['verdict'] == 'High'])
                    st.metric("High Relevance", high_relevance)
                with col3:
                    avg_score = round(df['relevance_score'].mean(), 2)
                    st.metric("Average Score", f"{avg_score}%")
                with col4:
                    top_score = round(df['relevance_score'].max(), 2)
                    st.metric("Top Score", f"{top_score}%")
                
                # Applications Table
                st.markdown("### Applications Overview")
                display_df = df[['student_name', 'email', 'relevance_score', 'hard_match_score', 
                               'semantic_score', 'verdict', 'evaluated_at']].copy()
                display_df.columns = ['Candidate Name', 'Email', 'Relevance Score (%)', 
                                    'Hard Match (%)', 'Semantic Score (%)', 'Verdict', 'Applied On']
                
                # Color code the dataframe
                def color_verdict(val):
                    if val == 'High': return 'background-color: #d4edda'
                    elif val == 'Medium': return 'background-color: #fff3cd'
                    else: return 'background-color: #f8d7da'
                
                styled_df = display_df.style.applymap(color_verdict, subset=['Verdict'])
                st.dataframe(styled_df, use_container_width=True)

                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Score Distribution Bar Chart
                    fig1 = px.bar(df, x="student_name", y="relevance_score", color="verdict",
                                  color_discrete_map={"High":"#28a745","Medium":"#ffc107","Low":"#dc3545"},
                                  title="Relevance Score by Candidate",
                                  labels={"student_name": "Candidate", "relevance_score": "Relevance Score (%)"})
                    fig1.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    # Verdict Distribution Pie Chart
                    verdict_counts = df['verdict'].value_counts()
                    fig_pie = px.pie(values=verdict_counts.values, names=verdict_counts.index,
                                     title="Verdict Distribution",
                                     color_discrete_map={"High":"#28a745","Medium":"#ffc107","Low":"#dc3545"})
                    st.plotly_chart(fig_pie, use_container_width=True)

                # Hard Match vs Semantic Score Scatter Plot
                df_numeric = df.copy()
                df_numeric['relevance_score'] = pd.to_numeric(df_numeric['relevance_score'], errors='coerce').fillna(1)
                df_numeric['hard_match_score'] = pd.to_numeric(df_numeric['hard_match_score'], errors='coerce').fillna(0)
                df_numeric['semantic_score'] = pd.to_numeric(df_numeric['semantic_score'], errors='coerce').fillna(0)
                df_numeric['size_score'] = df_numeric['relevance_score'].apply(lambda x: max(x/2, 10))

                fig2 = px.scatter(df_numeric, x="hard_match_score", y="semantic_score",
                                  size="size_score", color="verdict", hover_name="student_name",
                                  color_discrete_map={"High":"#28a745","Medium":"#ffc107","Low":"#dc3545"},
                                  title="Hard Match vs Semantic Score Analysis",
                                  labels={"hard_match_score": "Hard Match Score (%)", 
                                         "semantic_score": "Semantic Score (%)"})
                st.plotly_chart(fig2, use_container_width=True)

                # Missing Skills Analysis
                st.markdown("### Missing Skills Analysis")
                all_missing_skills = []
                for eval_item in evaluations:
                    if eval_item['missing_skills']:
                        all_missing_skills.extend(eval_item['missing_skills'])
                
                if all_missing_skills:
                    from collections import Counter
                    skill_counts = Counter(all_missing_skills)
                    skills_df = pd.DataFrame(list(skill_counts.items()), columns=['Skill', 'Count'])
                    skills_df = skills_df.sort_values('Count', ascending=False).head(10)
                    
                    fig3 = px.bar(skills_df, x='Count', y='Skill', orientation='h',
                                  title="Most Commonly Missing Skills",
                                  labels={'Count': 'Number of Candidates Missing This Skill'})
                    fig3.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.success(" Great! All candidates have the required skills!")

                # Detailed candidate information
                st.markdown("### Detailed Candidate Information")
                selected_candidate = st.selectbox("Select candidate for detailed view:", 
                                                ["-- Select Candidate --"] + df['student_name'].tolist())
                
                if selected_candidate != "-- Select Candidate --":
                    candidate_data = df[df['student_name'] == selected_candidate].iloc[0]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Name:** {candidate_data['student_name']}")
                        st.write(f"**Email:** {candidate_data['email']}")
                        st.write(f"**Applied On:** {candidate_data['evaluated_at']}")
                        st.write(f"**Overall Verdict:** {candidate_data['verdict']}")
                    
                    with col2:
                        st.write(f"**Relevance Score:** {candidate_data['relevance_score']}%")
                        st.write(f"**Hard Match Score:** {candidate_data['hard_match_score']}%")
                        st.write(f"**Semantic Score:** {candidate_data['semantic_score']}%")
                    
                    if candidate_data['missing_skills']:
                        st.write("**Missing Skills:**")
                        for skill in candidate_data['missing_skills']:
                            st.write(f"  - {skill}")
                    else:
                        st.success("No missing skills!")
                    
                    if candidate_data['suggestions']:
                        st.write("**Suggestions:**")
                        for suggestion in candidate_data['suggestions']:
                            st.write(f"  - {suggestion}")
            else:
                st.info(" No applications received for this job yet.")
    
    elif tab=="User":
        st.header("🔹 User Panel - Apply for Jobs")
        
        jobs = db.get_all_jobs()
        if not jobs:
            st.warning(" No jobs available at the moment. Please check back later.")
            return
        
        job_options = ["-- Select a Job to Apply --"]
        job_map = {}
        
        for job in jobs:
            option_text = f"{job.title} @ {job.company}"
            job_options.append(option_text)
            job_map[option_text] = job
        
        selected_option = st.selectbox("Select Job Position:", job_options)
        
        if selected_option != "-- Select a Job to Apply --":
            selected_job = job_map[selected_option]
            
            # Display job details
            with st.expander(f" Job Details: {selected_job.title}", expanded=True):
                st.write(f"**Company:** {selected_job.company}")
                st.write(f"**Must-have Skills:** {', '.join(selected_job.must_have_skills)}")
                if selected_job.good_to_have_skills:
                    st.write(f"**Good-to-have Skills:** {', '.join(selected_job.good_to_have_skills)}")
                if selected_job.education:
                    st.write(f"**Education Requirements:** {', '.join(selected_job.education)}")
                st.write(f"**Job Description:** {selected_job.description}")
            
            uploaded = st.file_uploader(" Upload Your Resume (PDF/DOCX)", 
                                      type=["pdf","docx"],
                                      help="Upload your resume in PDF or DOCX format for evaluation")
            
            if uploaded:
                try:
                    with st.spinner("Processing your resume..."):
                        bytes_content = uploaded.read()
                        file_type = uploaded.name.split(".")[-1]
                        resume = parser.parse_resume(bytes_content, file_type, uploaded.name)
                        
                        if resume is None:
                            st.error(" Failed to process resume. Please try a different file.")
                            return
                            
                        db.save_resume(resume)
                        evaluation = evaluator.evaluate(resume, selected_job)
                        db.save_evaluation(evaluation)
                    
                    st.success("Resume processed successfully!")
                    
                    # Results Section
                    st.markdown("---")
                    st.subheader(f" Evaluation Results for {resume.student_name}")
                    
                    # Score metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(" Relevance Score", f"{evaluation.relevance_score}%")
                    with col2:
                        st.metric(" Hard Match Score", f"{evaluation.hard_match_score}%")
                    with col3:
                        st.metric("Semantic Score", f"{evaluation.semantic_score}%")
                    with col4:
                        verdict_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
                        st.metric(" Verdict", f"{verdict_color.get(evaluation.verdict, '')} {evaluation.verdict}")
                    
                    # Visual Score Breakdown
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Score breakdown bar chart
                        score_df = pd.DataFrame({
                            "Score Type": ["Hard Match", "Semantic", "Overall Relevance"],
                            "Score": [evaluation.hard_match_score, evaluation.semantic_score, evaluation.relevance_score],
                            "Color": ["#17a2b8", "#28a745", "#6f42c1"]
                        })
                        
                        fig = px.bar(score_df, x="Score Type", y="Score", color="Color",
                                     color_discrete_map={color: color for color in score_df["Color"]},
                                     title=" Your Score Breakdown",
                                     labels={"Score": "Score (%)"})
                        fig.update_layout(showlegend=False, yaxis=dict(range=[0, 100]))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Gauge chart for overall score
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = evaluation.relevance_score,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Overall Score"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "#6f42c1"},
                                'steps': [
                                    {'range': [0, 50], 'color': "#f8d7da"},
                                    {'range': [50, 75], 'color': "#fff3cd"},
                                    {'range': [75, 100], 'color': "#d4edda"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig_gauge.update_layout(height=300)
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Detailed Feedback Section
                    st.markdown("###  Detailed Feedback")
                    
                    # Missing Skills
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("####  Missing Skills")
                        if evaluation.missing_skills:
                            for skill in evaluation.missing_skills:
                                st.markdown(f" **{skill}**")
                        else:
                            st.success("All required skills found!")
                    
                    with col2:
                        st.markdown("####  Missing Qualifications")
                        if evaluation.missing_qualifications:
                            for qual in evaluation.missing_qualifications:
                                st.markdown(f" **{qual}**")
                        else:
                            st.success("Education requirements met!")
                    
                    # Suggestions
                    st.markdown("####  Recommendations for Improvement")
                    if evaluation.suggestions:
                        for i, suggestion in enumerate(evaluation.suggestions, 1):
                            st.markdown(f"**{i}.** {suggestion}")
                    else:
                        st.success(" Your resume looks great for this position!")
                    
                    # Extracted Information
                    with st.expander(" Extracted Information from Your Resume", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Detected Skills:**")
                            if resume.skills:
                                for skill in resume.skills:
                                    st.write(f"  • {skill}")
                            else:
                                st.write("  No skills detected")
                                
                            st.write("**Contact Information:**")
                            st.write(f"  • Email: {resume.email if resume.email else 'Not found'}")
                            st.write(f"  • Phone: {resume.phone if resume.phone else 'Not found'}")
                        
                        with col2:
                            st.write("**Education:**")
                            if resume.education:
                                for edu in resume.education:
                                    st.write(f"  • {edu}")
                            else:
                                st.write("  No education information detected")
                    
                    # Next Steps
                    st.markdown("###  Next Steps")
                    if evaluation.verdict == "High":
                        st.success(" Congratulations! Your resume shows strong relevance for this position. You're likely to be a good fit!")
                    elif evaluation.verdict == "Medium":
                        st.warning("Your resume shows moderate relevance. Consider addressing the missing skills and suggestions above to strengthen your application.")
                    else:
                        st.error("Your resume needs improvement for this position. Focus on developing the missing skills and qualifications mentioned above.")
                    
                    # Download option (if needed)
                    st.markdown("---")
                    st.info(" Your evaluation has been saved. You can apply to more positions or return later to check your results.")
                
                except Exception as e:
                    st.error(f"Error processing resume: {str(e)}")
                    st.info("Please try uploading a different file or contact support if the issue persists.")

if __name__=="__main__":
    main()
