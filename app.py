import streamlit as st
import PyPDF2
import re
import nltk
import mlflow
import numpy as np
import spacy
import time
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords

# INITIAL SETUP: LOAD MODELS, STOPWORDS, AND TRACKING CONFIG

# Download stopwords once; avoids repeated downloads on every rerun
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# Load SBERT for semantic similarity; optimized for speed and strong text similarity performance
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load spaCy's English model; fallback to a blank pipeline if deployment environment blocks large model download
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = spacy.blank("en")

# MLflow tracking setup; gracefully disables if a server isn't running
MODEL_NAME = "SBERT"
try:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("ResumeFit Matcher")
    MLFLOW_ENABLED = True
except Exception:
    MLFLOW_ENABLED = False

# STREAMLIT PAGE CONFIGURATION AND CUSTOM STYLING

st.set_page_config(page_title="ResumeFit Matcher", layout="centered")

st.markdown("""
    <style>
    .reportview-container {
        background: #0e1117 !important;
        color: #fafafa !important;
        font-family: "Segoe UI", sans-serif !important;
    }
    .stButton>button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-weight: bold !important;
        padding: 0.5em 1em !important;
        border-radius: 8px !important;
    }
    .stTextArea textarea {
        background-color: #1e1e1e !important;
        color: white !important;
        border-radius: 8px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Resume ↔ Job Matcher")
st.markdown("Upload or paste your **Resume** and **Job Description** below to check match compatibility.")

# SESSION STATE INITIALIZATION
# Persist user input, computation results, and flags across Streamlit reruns
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""
if "match_computed" not in st.session_state:
    st.session_state.match_computed = False
if "results" not in st.session_state:
    st.session_state.results = {}

# TEXT PROCESSING AND ANALYSIS FUNCTIONS

def preprocess(text: str) -> str:
    # Standardize text by lowercasing, removing punctuation, and filtering stopwords
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(tokens)

def get_embedding(text: str):
    # Generate SBERT embedding for the processed text
    clean_text = preprocess(text)
    return sbert_model.encode(clean_text)

def compute_similarity(vec1, vec2):
    # Cosine similarity to measure semantic closeness between two embeddings
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def extract_skills_ner(text: str):
    # Extract technical skills using a combination of spaCy NER and regex pattern matching
    doc = nlp(text)
    skills = set()

    # Add named entities (organizations, products, tech tools)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "FAC"]:
            skills.add(ent.text.lower())

    # Add regex-detected technical keywords and common programming libraries
    tech_keywords = re.findall(r"\b[a-zA-Z\+\#\d]+\b", text)
    for kw in tech_keywords:
        kw = kw.lower()
        if kw not in STOPWORDS and len(kw) > 1:
            if any(c.isdigit() for c in kw) or kw in [
                "python", "java", "c++", "c#", "sql", "pandas", "numpy",
                "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost",
                "azure", "aws", "databricks", "bert", "gpt", "transformers"
            ]:
                skills.add(kw)
    return skills

def keyword_overlap(resume: str, jd: str) -> float:
    # Percentage of job description technical keywords also found in the resume
    resume_skills = extract_skills_ner(resume)
    jd_skills = extract_skills_ner(jd)
    if not jd_skills:
        return 0.0
    return len(resume_skills & jd_skills) / len(jd_skills)

def compute_final_score(embedding_score: float, keyword_score: float, alpha: float = 0.5):
    # Weighted combination of semantic similarity and keyword overlap for a holistic match score
    return alpha * embedding_score + (1 - alpha) * keyword_score

def extract_text_from_pdf(file):
    # Extracts text from PDFs with a real-time progress bar for user feedback
    try:
        text = ""
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)
        progress = st.progress(0, text="Extracting text from PDF...")
        for idx, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text() or ""
            text += page_text
            progress.progress(int(((idx + 1) / total_pages) * 100), text=f"Processing page {idx + 1}/{total_pages}...")
        time.sleep(0.3)
        progress.empty()  # remove the progress bar after completion
        return text.strip()
    except Exception as e:
        st.error(f"Could not read PDF: {e}. Please paste the text manually.")
        return ""

def generate_feedback(resume: str, jd: str, embedding_score: float, keyword_score: float):
    # Generates actionable feedback highlighting missing skills and alignment strengths
    resume_skills = extract_skills_ner(resume)
    jd_skills = extract_skills_ner(jd)
    missing = jd_skills - resume_skills

    feedback = []
    if embedding_score < 0.5:
        feedback.append(f"Your resume and job description have low semantic similarity (Embedding Score: {embedding_score:.2f}).")
    if keyword_score < 0.5:
        if missing:
            feedback.append(f"Consider adding these missing skills: {', '.join(missing)}.")
        else:
            feedback.append("Few technical keywords match; consider rephrasing skills to align better.")
    if not feedback:
        feedback.append("Strong alignment detected between your resume and the job description.")
    return "\n".join(feedback)

# USER INTERFACE: FILE UPLOADS AND TEXT INPUT

col1, col2 = st.columns(2)
with col1:
    st.subheader("Upload Resume (PDF)")
    resume_file = st.file_uploader("Upload Resume", type=["pdf"], key="resume_upload")

with col2:
    st.subheader("Upload Job Description (PDF)")
    jd_file = st.file_uploader("Upload JD", type=["pdf"], key="jd_upload")

st.subheader("OR Paste Text Instead")
resume_input = st.text_area("Paste Resume Text", height=180, disabled=bool(resume_file))
jd_input = st.text_area("Paste Job Description Text", height=180, disabled=bool(jd_file))

# Ensure only one input method (PDF or text) per field
if (resume_file and resume_input.strip()) or (jd_file and jd_input.strip()):
    st.error("Please use either PDF upload OR text input for each field, not both.")
    st.stop()

# Update session state based on current input
if resume_file:
    st.session_state.resume_text = extract_text_from_pdf(resume_file)
elif resume_input.strip():
    st.session_state.resume_text = resume_input.strip()

if jd_file:
    st.session_state.jd_text = extract_text_from_pdf(jd_file)
elif jd_input.strip():
    st.session_state.jd_text = jd_input.strip()

# COMPUTE MATCH SCORE ON BUTTON CLICK WITH REAL-TIME PROGRESS

if st.button("Compute Match Score"):
    if not st.session_state.resume_text or not st.session_state.jd_text:
        st.warning("Please provide both resume and job description.")
    else:
        progress = st.progress(0, text="Starting computation...")
        time.sleep(0.2)

        # Step 1: Generate embeddings for both texts
        progress.progress(30, text="Generating embeddings for resume and job description...")
        resume_vec = get_embedding(preprocess(st.session_state.resume_text))
        jd_vec = get_embedding(preprocess(st.session_state.jd_text))
        time.sleep(0.3)

        # Step 2: Compute semantic similarity
        progress.progress(60, text="Calculating semantic similarity...")
        embedding_score = compute_similarity(resume_vec, jd_vec)
        time.sleep(0.3)

        # Step 3: Compute keyword overlap
        progress.progress(80, text="Analyzing technical keyword overlap...")
        overlap_score = keyword_overlap(st.session_state.resume_text, st.session_state.jd_text)
        time.sleep(0.3)

        # Step 4: Combine into final score
        progress.progress(95, text="Finalizing match score...")
        final_score = compute_final_score(embedding_score, overlap_score)
        time.sleep(0.3)

        progress.progress(100, text="Computation complete!")
        time.sleep(0.5)
        progress.empty()

        if MLFLOW_ENABLED:
            mlflow.log_param("model", MODEL_NAME)
            mlflow.log_param("resume_length", len(st.session_state.resume_text))
            mlflow.log_param("jd_length", len(st.session_state.jd_text))
            mlflow.log_metric("embedding_score", embedding_score)
            mlflow.log_metric("keyword_overlap", overlap_score)
            mlflow.log_metric("final_match_score", final_score)

        st.session_state.results = {
            "final_score": final_score,
            "embedding_score": embedding_score,
            "keyword_score": overlap_score,
            "feedback": generate_feedback(st.session_state.resume_text, st.session_state.jd_text, embedding_score, overlap_score),
            "overlapping_keywords": extract_skills_ner(st.session_state.resume_text) & extract_skills_ner(st.session_state.jd_text)
        }
        st.session_state.match_computed = True

# DISPLAY RESULTS AFTER COMPUTATION

if st.session_state.match_computed:
    st.success(f"Match Score: {st.session_state.results['final_score'] * 100:.2f}%")

    st.markdown("#### Overlapping Keywords:")
    if st.session_state.results["overlapping_keywords"]:
        highlighted = ", ".join(f"`{word}`" for word in sorted(st.session_state.results["overlapping_keywords"]))
        st.markdown(
            f"<div style='color: lightgreen !important; font-weight: bold !important'>{highlighted}</div>",
            unsafe_allow_html=True
        )
    else:
        st.info("No overlapping technical keywords found. Add relevant tools or technologies.")

    st.markdown("#### Why You Got This Score:")
    st.markdown(
        f"<div style='color: gold !important; background-color:#1e1e1e !important; padding:10px; border-radius:5px'>"
        f"{st.session_state.results['feedback']}</div>",
        unsafe_allow_html=True
    )

    st.download_button("Download Feedback Report", st.session_state.results["feedback"], file_name="resume_match_feedback.txt")