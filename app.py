import streamlit as st
import fitz  # PyMuPDF
import os
import re
import nltk
import mlflow
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords

# Download stopwords if not already
nltk.download('stopwords')

# Load English stopwords
STOPWORDS = set(stopwords.words('english'))

# Load SBERT model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
MODEL_NAME = "SBERT"

# Set MLflow tracking URI (can be local or remote)
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Update this as needed
mlflow.set_experiment("ResumeFit Matcher")

# Streamlit page configuration
st.set_page_config(page_title="ResumeFit Matcher", layout="centered")

# Streamlit CSS style customization
st.markdown("""
    <style>
    .reportview-container {
        background: #f9f9f9;
        color: #333;
        font-family: "Segoe UI", sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5em 1em;
        border-radius: 8px;
    }
    .stTextArea textarea {
        background-color: #fffdf9;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Resume ↔ Job Matcher")
st.markdown("Upload or paste your **Resume** and **Job Description** below to check match compatibility.")

# Preprocess text: lowercase, remove punctuation, remove stopwords
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)

# Get SBERT embedding
def get_embedding(text: str):
    clean_text = preprocess(text)
    return sbert_model.encode(clean_text)

# Compute cosine similarity
def compute_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

# Compute keyword overlap
def keyword_overlap(resume: str, jd: str) -> float:
    resume_tokens = set(resume.split())
    jd_tokens = set(jd.split())
    if not jd_tokens:
        return 0.0
    return len(resume_tokens & jd_tokens) / len(jd_tokens)

# Final weighted score (alpha = embedding weight)
def compute_final_score(embedding_score, keyword_score, alpha=0.7):
    return alpha * embedding_score + (1 - alpha) * keyword_score

# Extract plain text from PDF using PyMuPDF
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

# PDF upload UI
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Resume (PDF)")
    resume_file = st.file_uploader("Upload Resume", type=["pdf"], key="resume_upload")

with col2:
    st.subheader("Upload Job Description (PDF)")
    jd_file = st.file_uploader("Upload JD", type=["pdf"], key="jd_upload")

# Text fallback fields
st.subheader("OR Paste Text Instead")
resume_input = st.text_area("Paste Resume Text", height=180)
jd_input = st.text_area("Paste Job Description Text", height=180)

# Extract content from PDFs if uploaded
resume_text = extract_text_from_pdf(resume_file) if resume_file else resume_input
jd_text = extract_text_from_pdf(jd_file) if jd_file else jd_input

# Compute Match Score
if st.button("Compute Match Score"):
    if not resume_text or not jd_text:
        st.warning("Please provide both resume and job description.")
    else:
        # Run MLflow logging and scoring
        with mlflow.start_run():
            # Preprocess text
            resume_clean = preprocess(resume_text)
            jd_clean = preprocess(jd_text)

            # Compute embeddings
            resume_vec = get_embedding(resume_clean)
            jd_vec = get_embedding(jd_clean)
            embedding_score = compute_similarity(resume_vec, jd_vec)

            # Compute keyword overlap
            overlap_score = keyword_overlap(resume_clean, jd_clean)

            # Final match score
            final_score = compute_final_score(embedding_score, overlap_score)

            # Log metadata and metrics to MLflow
            mlflow.log_param("model", MODEL_NAME)
            mlflow.log_param("resume_length", len(resume_text))
            mlflow.log_param("jd_length", len(jd_text))
            mlflow.log_metric("embedding_score", embedding_score)
            mlflow.log_metric("keyword_overlap", overlap_score)
            mlflow.log_metric("final_match_score", final_score)

            # Display results
            st.success(f"Match Score: {final_score * 100:.2f}%")

            st.markdown("#### Overlapping Keywords:")
            overlapping_words = list(set(resume_clean.split()) & set(jd_clean.split()))
            if overlapping_words:
                highlighted = ", ".join(f"`{word}`" for word in sorted(overlapping_words))
                st.markdown(f"<div style='color: green; font-weight: bold'>{highlighted}</div>", unsafe_allow_html=True)
            else:
                st.info("No overlapping keywords found. Try aligning your resume with the job description.")