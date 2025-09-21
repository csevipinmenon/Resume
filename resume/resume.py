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
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Document parsing
try:
    import pdfplumber
    import docx2txt
    DOC_LIBS_AVAILABLE = True
except:
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
        c.execute("SELECT * FROM jobs")
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
    
    def get_evaluations_by_job(self, job_id: str) -> List[dict]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            SELECT e.*, r.student_name, r.email FROM evaluations e
            JOIN resumes r ON e.resume_id = r.resume_id
            WHERE job_id = ? ORDER BY relevance_score DESC
        """, (job_id,))
        rows = c.fetchall()
        conn.close()
        evals = []
        for r in rows:
            evals.append({
                "evaluation_id": r[0],
                "resume_id": r[1],
                "job_id": r[2],
                "relevance_score": r[3],
                "hard_match_score": r[4],
                "semantic_score": r[5],
                "verdict": r[6],
                "missing_skills": json.loads(r[7]),
                "missing_qualifications": json.loads(r[8]),
                "suggestions": json.loads(r[9]),
                "evaluated_at": r[10],
                "student_name": r[11],
                "email": r[12]
            })
        return evals

# ------------------- DOCUMENT PARSER -------------------
class DocumentParser:
    def extract_text_from_pdf(self, content: bytes) -> str:
        try:
            import io
            text = ""
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for p in pdf.pages: text += (p.extract_text() or "") + "\n"
            return text
        except: return ""
    
    def extract_text_from_docx(self, content: bytes) -> str:
        try:
            import io
            return docx2txt.process(io.BytesIO(content))
        except: return ""
    
    def clean_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()
    
    def extract_email(self, text: str) -> str:
        m = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text)
        return m[0] if m else ""
    
    def extract_phone(self, text: str) -> str:
        m = re.findall(r'\+?\d[\d\s\-]{7,}\d', text)
        return m[0] if m else ""
    
    def extract_skills(self, text: str) -> List[str]:
        keywords = ["python","java","javascript","c++","aws","docker","sql","react","node.js"]
        return [kw.title() for kw in keywords if kw in text.lower()]
    
    def extract_education(self, text: str) -> List[str]:
        keywords = ["bachelor","master","phd"]
        return [kw.title() for kw in keywords if kw in text.lower()]
    
    def parse_resume(self, content: bytes, file_type: str, filename: str="") -> ResumeData:
        raw_text = ""
        if file_type.lower() == "pdf": raw_text = self.extract_text_from_pdf(content)
        elif file_type.lower() in ["docx","doc"]: raw_text = self.extract_text_from_docx(content)
        raw_text = self.clean_text(raw_text)
        resume_id = hashlib.md5(raw_text.encode()).hexdigest()[:12]
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

# ------------------- RESUME EVALUATOR -------------------
class ResumeEvaluator:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2') if SENTENCE_TRANSFORMERS_AVAILABLE else None
    
    def calc_hard_score(self, resume: ResumeData, job: JobRequirement) -> Tuple[float,List[str]]:
        total=0; max_score=0; missing=[]
        resume_skills=[s.lower() for s in resume.skills]
        for s in job.must_have_skills:
            max_score += 0.7
            if s.lower() in resume_skills: total+=0.7
            else: missing.append(s)
        for s in job.good_to_have_skills:
            max_score +=0.3
            if s.lower() in resume_skills: total+=0.3
        max_score +=0.3
        if any(e.lower() in ' '.join(resume.education).lower() for e in job.education):
            total+=0.3
        return round((total/max_score*100 if max_score else 0),2), missing
    
    def calc_semantic_score(self,resume:ResumeData,job:JobRequirement)->float:
        if self.model:
            try:
                r_text=f"{' '.join(resume.skills)} {' '.join(resume.education)} {resume.raw_text[:2000]}"
                j_text=f"{job.title} {' '.join(job.must_have_skills)} {' '.join(job.good_to_have_skills)} {job.description[:2000]}"
                r_emb=self.model.encode([r_text])
                j_emb=self.model.encode([j_text])
                return round(float(cosine_similarity(r_emb,j_emb)[0][0]*100),2)
            except: return 30.0
        return 30.0
    
    def generate_verdict(self,score:float)->str:
        if score>=75:return "High"
        elif score>=50:return "Medium"
        else: return "Low"
    
    def generate_suggestions(self,missing_skills:List[str],resume:ResumeData)->List[str]:
        sug=[]
        if missing_skills: sug.append(f"Learn or highlight these missing skills: {', '.join(missing_skills)}")
        if not resume.skills: sug.append("Add 1-2 relevant projects showing hands-on experience.")
        return sug
    
    def evaluate(self,resume:ResumeData,job:JobRequirement)->EvaluationResult:
        hard_score, missing_skills=self.calc_hard_score(resume,job)
        semantic_score=self.calc_semantic_score(resume,job)
        relevance_score=round(hard_score*0.6+semantic_score*0.4,2)
        verdict=self.generate_verdict(relevance_score)
        missing_qual=[]
        for edu in job.education:
            if edu.lower() not in ' '.join(resume.education).lower(): missing_qual.append(edu)
        suggestions=self.generate_suggestions(missing_skills,resume)
        eval_id=hashlib.md5(f"{resume.resume_id}_{job.job_id}".encode()).hexdigest()[:12]
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
    st.set_page_config("Automated Resume Evaluation", layout="wide",)
    st.title("Automated Resume Relevance Check System")
    
    db = DatabaseManager()
    parser = DocumentParser()
    evaluator = ResumeEvaluator()
    
    tab = st.sidebar.radio("Select Mode", ["User", "Admin"])
    
    if tab=="Admin":
        st.header("🔹 Admin Panel")
        with st.expander("Add Job Description"):
            title = st.text_input("Job Title")
            company = st.text_input("Company")
            desc = st.text_area("Job Description")
            must_skills = st.text_input("Must-have Skills (comma)").split(",")
            good_skills = st.text_input("Good-to-have Skills (comma)").split(",")
            education = st.text_input("Education (comma)").split(",")
            if st.button("Save Job"):
                job = JobRequirement(
                    job_id=hashlib.md5(desc.encode()).hexdigest()[:12],
                    title=title, company=company, description=desc,
                    must_have_skills=[s.strip() for s in must_skills if s.strip()],
                    good_to_have_skills=[s.strip() for s in good_skills if s.strip()],
                    education=[e.strip() for e in education if e.strip()],
                    created_at=datetime.now()
                )
                db.save_job(job)
                st.success(f"Job '{title}' saved!")
        
        jobs = db.get_all_jobs()
        job_map = {f"{j.title}@{j.company}": j for j in jobs}
        choice = st.selectbox("Select Job", [""] + list(job_map.keys()))
        
        if choice:
            evaluations = db.get_evaluations_by_job(job_map[choice].job_id)
            if evaluations:
                df = pd.DataFrame(evaluations)
                st.markdown("### Evaluation Table")
                st.dataframe(df[["student_name","email","relevance_score","verdict"]])
                
                # Score Distribution Bar
                fig1 = px.bar(df, x="student_name", y="relevance_score", color="verdict",
                              color_discrete_map={"High":"green","Medium":"orange","Low":"red"},
                              title="Relevance Score per Candidate")
                st.plotly_chart(fig1, use_container_width=True)
                
                # Hard vs Semantic Scatter
                fig2 = px.scatter(df, x="hard_match_score", y="semantic_score", 
                                  size="relevance_score", color="verdict", hover_name="student_name",
                                  color_discrete_map={"High":"green","Medium":"orange","Low":"red"},
                                  title="Hard Match vs Semantic Score")
                st.plotly_chart(fig2, use_container_width=True)
                
                # Missing Skills Pie
                st.markdown("### Missing Skills Overview")
                all_missing = sum(df['missing_skills'].tolist(), [])
                if all_missing:
                    ms_df = pd.DataFrame({"skill": all_missing})
                    ms_count = ms_df['skill'].value_counts().reset_index()
                    ms_count.columns = ["Skill", "Count"]
                    fig3 = px.pie(ms_count, names="Skill", values="Count", title="Most Missing Skills")
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("All candidates have required skills")
            else:
                st.info("No evaluations yet.")
    
    elif tab=="User":
        st.header("🔹 User Panel")
        jobs = db.get_all_jobs()
        job_map = {f"{j.title}@{j.company}": j for j in jobs}
        choice = st.selectbox("Select Job to Apply", [""] + list(job_map.keys()))
        uploaded = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf","docx"])
        
        if choice and uploaded:
            bytes_content = uploaded.read()
            file_type = uploaded.name.split(".")[-1]
            resume = parser.parse_resume(bytes_content, file_type, uploaded.name)
            db.save_resume(resume)
            evaluation = evaluator.evaluate(resume, job_map[choice])
            db.save_evaluation(evaluation)
            
            st.subheader(f"Evaluation Result for {resume.student_name}")
            col1, col2 = st.columns(2)
            
            # Metrics
            col1.metric(" Relevance Score", f"{evaluation.relevance_score}%")
            col1.metric(" Hard Match Score", f"{evaluation.hard_match_score}%")
            col1.metric(" Semantic Score", f"{evaluation.semantic_score}%")
            col1.write(f"**Verdict:** {evaluation.verdict}")
            
            # Graphs
            score_df = pd.DataFrame({
                "Type": ["Hard Match","Semantic","Relevance"],
                "Score":[evaluation.hard_match_score, evaluation.semantic_score, evaluation.relevance_score]
            })
            fig = px.bar(score_df, x="Type", y="Score", color="Score",
                         color_continuous_scale="Viridis", text="Score", title="Score Breakdown")
            col2.plotly_chart(fig, use_container_width=True)
            
            # Missing Skills
            st.markdown("### Missing Skills")
            if evaluation.missing_skills:
                for s in evaluation.missing_skills: st.markdown(f"- {s}")
            else: st.success("No missing skills!")
            
            # Missing Qualifications
            st.markdown("###  Missing Qualifications")
            if evaluation.missing_qualifications:
                for q in evaluation.missing_qualifications: st.markdown(f"- {q}")
            else: st.success("No missing qualifications!")
            
            # Suggestions
            st.markdown("###  Suggestions")
            for s in evaluation.suggestions: st.markdown(f"- {s}")

if __name__=="__main__":
    main()
