import streamlit as st
import fitz
import os
import re
import nltk
import mlflow
import numpy as np
import io
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from datetime import datetime
import pandas as pd
from mlflow.tracking import MlflowClient

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# Load embedding model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
MODEL_NAME = "SBERT"

# Setup MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("ResumeFit Matcher")

# Streamlit config
st.set_page_config(page_title="ResumeFit Matcher", layout="wide")

st.title("Resume ↔ Job Matcher")
st.markdown("Upload your **Resume** and **Job Description** as PDFs or paste the text manually to analyze semantic match score.")

# Helper: Preprocessing
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)

# Helper: Extract PDF text
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

# Helper: Embedding
def get_embedding(text: str):
    return sbert_model.encode(preprocess(text))

# Helper: Cosine similarity
def compute_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

# Helper: Keyword overlap
def keyword_overlap(resume: str, jd: str) -> float:
    resume_tokens = set(preprocess(resume).split())
    jd_tokens = set(preprocess(jd).split())
    if not jd_tokens:
        return 0.0
    return len(resume_tokens & jd_tokens) / len(jd_tokens)

# Final scoring
def compute_final_score(embedding_score, keyword_score, alpha=0.7):
    return alpha * embedding_score + (1 - alpha) * keyword_score

# Feedback explanation
def get_feedback(embedding_score, keyword_score, final_score):
    tips = []
    if embedding_score < 0.6:
        tips.append("Semantic match is weak. Consider aligning your language with job keywords.")
    if keyword_score < 0.3:
        tips.append("Few shared keywords. Tailor your resume to match the job description.")
    if final_score < 0.5:
        tips.append("Overall match is low. Try emphasizing relevant experience and terminology.")
    if not tips:
        tips.append("Great alignment between your resume and the job description.")
    return tips

# PDF Upload or Text
col1, col2 = st.columns(2)
with col1:
    st.subheader("Upload Resume (PDF)")
    resume_file = st.file_uploader("Upload Resume", type=["pdf"], key="resume")
with col2:
    st.subheader("Upload Job Description (PDF)")
    jd_file = st.file_uploader("Upload JD", type=["pdf"], key="jd")

st.subheader("OR Paste Text")
resume_text_input = st.text_area("Paste Resume Text", height=150)
jd_text_input = st.text_area("Paste Job Description Text", height=150)

resume_text = extract_text_from_pdf(resume_file) if resume_file else resume_text_input
jd_text = extract_text_from_pdf(jd_file) if jd_file else jd_text_input

if st.button("Compute Match Score"):
    if not resume_text or not jd_text:
        st.warning("Please upload or paste both resume and job description.")
    else:
        with mlflow.start_run():
            resume_clean = preprocess(resume_text)
            jd_clean = preprocess(jd_text)
            resume_vec = get_embedding(resume_clean)
            jd_vec = get_embedding(jd_clean)

            emb_score = compute_similarity(resume_vec, jd_vec)
            kw_score = keyword_overlap(resume_clean, jd_clean)
            final_score = compute_final_score(emb_score, kw_score)

            overlap_words = sorted(list(set(resume_clean.split()) & set(jd_clean.split())))
            feedback = get_feedback(emb_score, kw_score, final_score)

            mlflow.log_param("model", MODEL_NAME)
            mlflow.log_param("resume_length", len(resume_text))
            mlflow.log_param("jd_length", len(jd_text))
            mlflow.log_metric("embedding_score", emb_score)
            mlflow.log_metric("keyword_overlap", kw_score)
            mlflow.log_metric("final_match_score", final_score)

            st.success(f"Match Score: {final_score * 100:.2f}%")
            st.markdown("#### Overlapping Keywords:")
            if overlap_words:
                st.markdown(", ".join(f"`{word}`" for word in overlap_words))
            else:
                st.info("No overlapping keywords found.")

            st.markdown("#### Feedback Suggestions:")
            for item in feedback:
                st.write(f"- {item}")

            # Downloadable report
            report = f"""
            ResumeFit Report – {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

            Match Score: {final_score * 100:.2f}%
            Embedding Score: {emb_score:.4f}
            Keyword Overlap: {kw_score:.4f}
            Overlapping Keywords: {", ".join(overlap_words)}

            Feedback:
            {' | '.join(feedback)}
            """
            st.download_button("Download Report", data=report, file_name="resumefit_report.txt")

# MLflow Analytics Dashboard
st.markdown("---")
st.subheader("MLflow Dashboard (All User Trends)")

if st.button("Show Trend Analytics"):
    client = MlflowClient()
    runs = client.search_runs(experiment_ids=["0"])
    data = []
    for r in runs:
        m = r.data.metrics
        p = r.data.params
        data.append({
            "Embedding Score": m.get("embedding_score", 0),
            "Keyword Overlap": m.get("keyword_overlap", 0),
            "Final Score": m.get("final_match_score", 0),
            "Resume Length": p.get("resume_length", 0),
            "JD Length": p.get("jd_length", 0),
        })
    df = pd.DataFrame(data)
    st.dataframe(df)

    st.markdown("##### Score Distribution")
    st.line_chart(df[["Embedding Score", "Keyword Overlap", "Final Score"]])