##  Features

### Admin Panel
- Add and manage job descriptions.
- View all evaluations for a specific job.
- Visual analytics:
  - Relevance score bar charts.
  - Hard vs semantic score scatter plots.
  - Missing skills pie charts.
- Download evaluation reports for further analysis.

### User Panel
- Upload PDF or DOCX resumes.
- Automatic extraction of:
  - Name, Email, Phone
  - Skills and Education
- Resume evaluation against selected job:
  - Hard match score (skills + education)
  - Semantic score using SentenceTransformer embeddings
  - Relevance score (combined)
  - Verdict: High / Medium / Low
- Graphical visualization of scores.
- Suggestions for missing skills and qualifications.

---

##  Tech Stack

- **Python 3.10+**
- **Streamlit** for UI
- **SQLite** for database
- **Plotly** for interactive charts
- **pdfplumber / docx2txt** for resume parsing
- **SentenceTransformers** for semantic evaluation
- **Pandas / JSON** for data processing


##  Workflows

### 1. Admin Workflow
1. Log in to the **Admin panel**.
2. Add a new **job description** with:
   - Title, Company, Description
   - Must-have skills, Good-to-have skills
   - Education requirements
3. View **evaluations** for each job:
   - Relevance score table
   - Graphical analysis
   - Insights on missing skills
4. Use data-driven insights to shortlist candidates.

**Visual Flow:**
```

\[Admin Panel] --> \[Add Job / Manage Jobs] --> \[View Evaluations] --> \[Graphical Insights]

```

### 2. User Workflow
1. Navigate to the **User panel**.
2. Select a job to apply for.
3. Upload your resume (PDF or DOCX).
4. System parses your resume and extracts:
   - Name, Email, Phone
   - Skills and Education
5. Resume is evaluated against the selected job:
   - Hard match score
   - Semantic score
   - Combined relevance score
6. Receive:
   - Score breakdown charts
   - Missing skills and qualifications
   - Actionable suggestions to improve resume

**Visual Flow:**
```

\[User Panel] --> \[Select Job] --> \[Upload Resume] --> \[Parsing & Evaluation] --> \[Score & Suggestions]

````

---

##  Visualizations

- **Bar Charts:** Individual relevance, hard, and semantic scores
- **Scatter Plot:** Hard vs semantic score comparison
- **Pie Charts:** Most missing skills among applicants
- **Metrics:** Display key evaluation numbers with badges for instant insight

---

##  Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/resume-evaluator.git
cd resume-evaluator
````

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```
```bash
streamlit run resume.py --server.fileWatcherType=none
```

---

##  Dependencies

```txt
streamlit
pandas
plotly
pdfplumber
docx2txt
sentence-transformers
scikit-learn
```

> **Note:** `sentence-transformers` is optional but recommended for semantic analysis.

---

##  Future Enhancements

* Multi-language resume support (French, German, etc.)
* Advanced skill matching using embeddings
* Export evaluation reports as PDF
* Role-based login for Admin and Users
* Integration with cloud storage for resume upload

---

##  Author

##Team - ZeroBug
**Saurabh Paswan**
**Priyaranjan**
**Vipin Kumar**

* Email: [vipinmeomon@gmail.com](mailto:vipinmemon8123@gmail.com)
* GitHub: [https://github.com/vipinmememon](https://github.com/vipinmememon)

---

##  License

MIT License © 2025

```

This **single-file README** includes:  
- Features, workflows, tech stack, visualizations  
- ASCII-style workflow diagrams for clarity  
- Installation, dependencies, future enhancements, and author/license info  

---

If you want, I can also **enhance it with small inline emoji-based charts and metrics examples** to make the README **look like a dashboard summary visually**.  

Do you want me to do that?
```
