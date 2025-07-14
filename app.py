import os
import re
import fitz  # PyMuPDF
import nltk
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords

# Try to load MLflow if available
try:
    import mlflow
    IS_LOCAL = os.getenv("IS_LOCAL", "false").lower() == "true"
    if IS_LOCAL:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("ResumeFit Matcher")
        ENABLE_MLFLOW = True
    else:
        ENABLE_MLFLOW = False
except ImportError:
    ENABLE_MLFLOW = False

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
MODEL_NAME = "SBERT"

st.set_page_config(page_title="ResumeFit Matcher", layout="centered")

st.markdown("""
    <style>
    .reportview-container {
        background: #f9f9f9;
        color: #333;
        font-family: "Segoe UI", sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white !important;
        font-weight: bold;
        padding: 0.5em 1em;
        border-radius: 8px;
    }
    .stTextArea textarea {
        background-color: #fffdf9 !important;
        color: #000000 !important;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Resume ↔ Job Matcher")
st.markdown("Upload or paste your **Resume** and **Job Description** below to check match compatibility.")

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)

def get_embedding(text: str):
    return sbert_model.encode(preprocess(text))

def compute_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def keyword_overlap(resume: str, jd: str) -> float:
    resume_tokens = set(resume.split())
    jd_tokens = set(jd.split())
    if not jd_tokens:
        return 0.0
    return len(resume_tokens & jd_tokens) / len(jd_tokens)

def compute_final_score(embedding_score, keyword_score, alpha=0.7):
    return alpha * embedding_score + (1 - alpha) * keyword_score

def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Upload Resume (PDF)")
    resume_file = st.file_uploader("Upload Resume", type=["pdf"], key="resume_upload")
with col2:
    st.subheader("Upload Job Description (PDF)")
    jd_file = st.file_uploader("Upload JD", type=["pdf"], key="jd_upload")

st.subheader("OR Paste Text Instead")
resume_input = st.text_area("Paste Resume Text", height=180)
jd_input = st.text_area("Paste Job Description Text", height=180)

resume_text = extract_text_from_pdf(resume_file) if resume_file else resume_input
jd_text = extract_text_from_pdf(jd_file) if jd_file else jd_input

if st.button("Compute Match Score"):
    if not resume_text or not jd_text:
        st.warning("Please provide both resume and job description.")
    else:
        resume_clean = preprocess(resume_text)
        jd_clean = preprocess(jd_text)

        resume_vec = get_embedding(resume_clean)
        jd_vec = get_embedding(jd_clean)

        embedding_score = compute_similarity(resume_vec, jd_vec)
        overlap_score = keyword_overlap(resume_clean, jd_clean)
        final_score = compute_final_score(embedding_score, overlap_score)

        if ENABLE_MLFLOW:
            with mlflow.start_run():
                mlflow.log_param("model", MODEL_NAME)
                mlflow.log_param("resume_length", len(resume_text))
                mlflow.log_param("jd_length", len(jd_text))
                mlflow.log_metric("embedding_score", embedding_score)
                mlflow.log_metric("keyword_overlap", overlap_score)
                mlflow.log_metric("final_match_score", final_score)

        st.success(f"Match Score: {final_score * 100:.2f}%")

        overlapping_words = sorted(set(resume_clean.split()) & set(jd_clean.split()))
        st.markdown("#### Overlapping Keywords:")
        if overlapping_words:
            st.markdown(f"<div style='color: green; font-weight: bold'>{', '.join(f'`{w}`' for w in overlapping_words)}</div>", unsafe_allow_html=True)
        else:
            st.info("No overlapping keywords found. Try aligning your resume with the job description.")

        st.markdown("### Why You Got This Score:")
        if embedding_score < 0.7:
            st.warning(f"Your resume and JD are semantically dissimilar (Embedding Score: {embedding_score:.2f}).")
        if overlap_score < 0.3:
            st.warning(f"Few matching keywords were found (Overlap Score: {overlap_score:.2f}).")
        if embedding_score >= 0.7 and overlap_score >= 0.3:
            st.success("Your resume aligns well both in meaning and keyword relevance.")

        st.markdown("### Downloadable Feedback")
        missing = sorted(set(jd_clean.split()) - set(resume_clean.split()))
        feedback = f"""
        Match Score: {final_score * 100:.2f}%
        Embedding Similarity: {embedding_score:.2f}
        Keyword Overlap: {overlap_score:.2f}
        Missing Keywords: {', '.join(missing[:50])}
        """
        st.download_button("Download Feedback Report", feedback, file_name="resume_match_feedback.txt")