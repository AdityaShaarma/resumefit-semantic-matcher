import streamlit as st
import fitz  # PyMuPDF for extracting text from PDF files
import re
import nltk
import mlflow
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords

# INITIAL SETUP: MODEL LOADING, STOPWORDS, AND TRACKING CONFIG

# Download stopwords if not already available
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# Load SBERT model for semantic similarity
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Ensure spaCy model is available; automatically download if missing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Model identifier for logging purposes
MODEL_NAME = "SBERT"

# Configure MLflow only for local runs; disable if server unavailable
try:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("ResumeFit Matcher")
    MLFLOW_ENABLED = True
except Exception:
    MLFLOW_ENABLED = False

# STREAMLIT PAGE CONFIGURATION AND CUSTOM STYLING

st.set_page_config(page_title="ResumeFit Matcher", layout="centered")

# Dark theme styling with enforced !important for consistent appearance
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

# Page title and description
st.title("Resume ↔ Job Matcher")
st.markdown("Upload or paste your **Resume** and **Job Description** below to check match compatibility.")

# TEXT PROCESSING AND ANALYSIS FUNCTIONS

def preprocess(text: str) -> str:
    """
    Preprocesses input text for semantic similarity calculations.
    - Converts to lowercase.
    - Removes punctuation and special characters.
    - Removes common English stopwords.
    Returns a cleaned string suitable for embedding generation.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)

def get_embedding(text: str):
    """
    Generates a vector representation of the text using SBERT.
    Returns a numpy array embedding.
    """
    clean_text = preprocess(text)
    return sbert_model.encode(clean_text)

def compute_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vector embeddings.
    Returns a value between 0 (no similarity) and 1 (perfect similarity).
    """
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def extract_skills_ner(text: str):
    """
    Extracts technical skills and keywords using:
    1. spaCy Named Entity Recognition (ORG, PRODUCT, etc.).
    2. Regex-based detection of common programming or ML terms.
    Returns a set of normalized lowercase skill terms.
    """
    doc = nlp(text)
    skills = set()

    # Named Entity Recognition-based extraction
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "FAC"]:
            skills.add(ent.text.lower())

    # Regex detection for common technical keywords and abbreviations
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
    """
    Calculates how many job description skills are present in the resume.
    Returns a proportion between 0 and 1.
    """
    resume_skills = extract_skills_ner(resume)
    jd_skills = extract_skills_ner(jd)
    if not jd_skills:
        return 0.0
    return len(resume_skills & jd_skills) / len(jd_skills)

def compute_final_score(embedding_score: float, keyword_score: float, alpha: float = 0.5):
    """
    Combines semantic similarity and keyword overlap into one final score.
    Alpha defines the weight given to semantic similarity.
    Returns a score between 0 and 1.
    """
    return alpha * embedding_score + (1 - alpha) * keyword_score

def extract_text_from_pdf(file):
    """
    Extracts text from all pages of a PDF file using PyMuPDF.
    Returns the full text as a single string.
    """
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

def generate_feedback(resume: str, jd: str, embedding_score: float, keyword_score: float):
    """
    Creates a human-readable explanation for the computed score.
    Includes:
    - Semantic similarity insights.
    - Missing technical skills compared to the job description.
    Returns feedback as a formatted string.
    """
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
            feedback.append("Few technical keywords match; consider rephrasing skills to match the job description.")
    if not feedback:
        feedback.append("Strong alignment detected between your resume and the job description.")

    return "\n".join(feedback)

# STREAMLIT USER INTERFACE AND ERROR HANDLING

# File upload options
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Resume (PDF)")
    resume_file = st.file_uploader("Upload Resume", type=["pdf"], key="resume_upload")

with col2:
    st.subheader("Upload Job Description (PDF)")
    jd_file = st.file_uploader("Upload JD", type=["pdf"], key="jd_upload")

# Text area fallback if PDFs are not uploaded
st.subheader("OR Paste Text Instead")
resume_input = st.text_area("Paste Resume Text", height=180, disabled=bool(resume_file))
jd_input = st.text_area("Paste Job Description Text", height=180, disabled=bool(jd_file))

# Determine input source (PDFs take priority)
resume_text = extract_text_from_pdf(resume_file) if resume_file else resume_input
jd_text = extract_text_from_pdf(jd_file) if jd_file else jd_input

# Warn user if they mix both inputs
if (resume_file and resume_input.strip()) or (jd_file and jd_input.strip()):
    st.warning("You uploaded a PDF and also typed text. Only the uploaded PDF will be used.")

# SCORE COMPUTATION AND DISPLAY

if st.button("Compute Match Score"):
    if not resume_text or not jd_text:
        st.warning("Please provide both resume and job description.")
    else:
        with mlflow.start_run() if MLFLOW_ENABLED else st.spinner("Computing match score..."):
            # Compute semantic similarity and keyword overlap
            resume_vec = get_embedding(preprocess(resume_text))
            jd_vec = get_embedding(preprocess(jd_text))
            embedding_score = compute_similarity(resume_vec, jd_vec)
            overlap_score = keyword_overlap(resume_text, jd_text)
            final_score = compute_final_score(embedding_score, overlap_score)

            # Log metrics locally if MLflow is active
            if MLFLOW_ENABLED:
                mlflow.log_param("model", MODEL_NAME)
                mlflow.log_param("resume_length", len(resume_text))
                mlflow.log_param("jd_length", len(jd_text))
                mlflow.log_metric("embedding_score", embedding_score)
                mlflow.log_metric("keyword_overlap", overlap_score)
                mlflow.log_metric("final_match_score", final_score)

            # Display final match score
            st.success(f"Match Score: {final_score * 100:.2f}%")

            # Display overlapping skills
            overlapping_keywords = extract_skills_ner(resume_text) & extract_skills_ner(jd_text)
            st.markdown("#### Overlapping Keywords:")
            if overlapping_keywords:
                highlighted = ", ".join(f"`{word}`" for word in sorted(overlapping_keywords))
                st.markdown(
                    f"<div style='color: lightgreen !important; font-weight: bold !important'>{highlighted}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.info("No overlapping technical keywords found. Add relevant tools or technologies.")

            # Display feedback reasoning
            st.markdown("#### Why You Got This Score:")
            feedback = generate_feedback(resume_text, jd_text, embedding_score, overlap_score)
            st.markdown(
                f"<div style='color: gold !important; background-color:#1e1e1e !important; "
                f"padding:10px; border-radius:5px'>{feedback}</div>",
                unsafe_allow_html=True
            )

            # Allow downloading the feedback as a text report
            st.download_button("Download Feedback Report", feedback, file_name="resume_match_feedback.txt")