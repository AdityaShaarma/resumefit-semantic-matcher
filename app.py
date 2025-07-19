import streamlit as st
import PyPDF2
import re
import nltk
import mlflow
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import time

# INITIAL SETUP: MODEL LOADING AND CONFIGURATION

# Download stopwords if they are not already present
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# Load SBERT model for semantic similarity with forced CPU usage
# Using device="cpu" avoids GPU-related NotImplementedError issues in Streamlit Cloud
sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Load or fallback to a blank spaCy model for NER and tokenization
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = spacy.blank("en")

# Model name reference for logging
MODEL_NAME = "SBERT"

# Configure MLflow for experiment tracking; gracefully disable if unavailable
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
st.markdown("Upload or paste your **Resume** and **Job Description** below to compute a compatibility score.")

# TEXT PROCESSING AND ANALYSIS FUNCTIONS

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)

def get_embedding(text: str):
    clean_text = preprocess(text)
    return sbert_model.encode(clean_text)

def compute_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def extract_skills_ner(text: str):
    doc = nlp(text)
    skills = set()

    # Named Entity Recognition-based skills
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "FAC"]:
            skills.add(ent.text.lower())

    # Keyword and regex-based detection of technical terms
    tech_keywords = re.findall(r"\b[a-zA-Z\+\#\d]+\b", text)
    for kw in tech_keywords:
        kw = kw.lower()
        if kw not in STOPWORDS and len(kw) > 1:
            skills.add(kw)
    return skills

def keyword_overlap(resume: str, jd: str) -> float:
    resume_skills = extract_skills_ner(resume)
    jd_skills = extract_skills_ner(jd)
    if not jd_skills:
        return 0.0
    return len(resume_skills & jd_skills) / len(jd_skills)

def compute_final_score(embedding_score: float, keyword_score: float, alpha: float = 0.5):
    return alpha * embedding_score + (1 - alpha) * keyword_score

def extract_text_from_pdf(file, progress_bar):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text() or ""
            text += page_text
            # Update progress bar incrementally
            progress_bar.progress((i + 1) / total_pages)
            time.sleep(0.05)
        return text.strip()
    except Exception as e:
        st.error(f"Could not read PDF: {e}. Please paste the text manually.")
        return ""

def generate_feedback(resume: str, jd: str, embedding_score: float, keyword_score: float):
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

# USER INPUT SECTION

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

if (resume_file and resume_input.strip()) or (jd_file and jd_input.strip()):
    st.error("Please use either PDF upload OR text input for each field, not both.")
    st.stop()

# PDF EXTRACTION WITH PROGRESS BAR ISOLATED AS A FRAGMENT

@st.experimental_fragment
def extract_text_with_progress():
    resume_text = ""
    jd_text = ""

    if resume_file:
        st.info("Extracting resume text...")
        progress = st.progress(0)
        resume_text = extract_text_from_pdf(resume_file, progress)
    else:
        resume_text = resume_input

    if jd_file:
        st.info("Extracting job description text...")
        progress = st.progress(0)
        jd_text = extract_text_from_pdf(jd_file, progress)
    else:
        jd_text = jd_input

    return resume_text, jd_text

resume_text, jd_text = extract_text_with_progress()

# MATCH SCORE COMPUTATION SECTION ISOLATED AS A FRAGMENT

@st.experimental_fragment
def compute_match_score_section():
    if st.button("Compute Match Score"):
        if not resume_text or not jd_text:
            st.warning("Please provide both resume and job description.")
            return

        st.info("Analyzing and computing match score...")
        progress = st.progress(0)
        total_steps = 5

        try:
            for step in range(total_steps):
                time.sleep(0.15)
                progress.progress((step + 1) / total_steps)

            resume_vec = get_embedding(preprocess(resume_text))
            jd_vec = get_embedding(preprocess(jd_text))
            embedding_score = compute_similarity(resume_vec, jd_vec)
            overlap_score = keyword_overlap(resume_text, jd_text)
            final_score = compute_final_score(embedding_score, overlap_score)

            if MLFLOW_ENABLED:
                with mlflow.start_run():
                    mlflow.log_param("model", MODEL_NAME)
                    mlflow.log_param("resume_length", len(resume_text))
                    mlflow.log_param("jd_length", len(jd_text))
                    mlflow.log_metric("embedding_score", embedding_score)
                    mlflow.log_metric("keyword_overlap", overlap_score)
                    mlflow.log_metric("final_match_score", final_score)

            st.success(f"Match Score: {final_score * 100:.2f}%")

            st.markdown("#### Overlapping Keywords:")
            overlapping_keywords = extract_skills_ner(resume_text) & extract_skills_ner(jd_text)
            if overlapping_keywords:
                highlighted = ", ".join(f"`{word}`" for word in sorted(overlapping_keywords))
                st.markdown(
                    f"<div style='color: lightgreen; font-weight: bold'>{highlighted}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.info("No overlapping technical keywords found. Add relevant tools or technologies.")

            st.markdown("#### Why You Got This Score:")
            feedback = generate_feedback(resume_text, jd_text, embedding_score, overlap_score)
            st.markdown(
                f"<div style='color: gold; background-color:#1e1e1e; padding:10px; border-radius:5px'>{feedback}</div>",
                unsafe_allow_html=True
            )

            st.download_button("Download Feedback Report", feedback, file_name="resume_match_feedback.txt")

        except Exception as e:
            st.error(f"An error occurred while computing the match score: {e}")

compute_match_score_section()