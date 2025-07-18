import streamlit as st
import PyPDF2  # For extracting text from PDF files
import re
import nltk
import mlflow
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import time  # For simulating progress bar updates

# INITIAL SETUP: LOAD MODELS, STOPWORDS, AND TRACKING CONFIG

# Download stopwords if not already downloaded; avoids errors on first run
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# Load the SBERT model for semantic similarity (compact but effective)
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Ensure spaCy English model is available; fallback to blank English model if download fails
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = spacy.blank("en")

# Model name for logging and tracking
MODEL_NAME = "SBERT"

# Attempt to connect to local MLflow server; disable logging if unavailable
try:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("ResumeFit Matcher")
    MLFLOW_ENABLED = True
except Exception:
    MLFLOW_ENABLED = False

# STREAMLIT PAGE CONFIGURATION AND CUSTOM STYLING

st.set_page_config(page_title="ResumeFit Matcher", layout="centered")

# Custom dark theme styling for consistency across browsers
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

# Page header and brief instructions for users
st.title("Resume ↔ Job Matcher")
st.markdown("Upload or paste your **Resume** and **Job Description** below to check match compatibility.")

# TEXT PROCESSING AND ANALYSIS FUNCTIONS

def preprocess(text: str) -> str:
    # Clean and standardize text for semantic similarity:
    # 1. lowercase everything
    # 2. remove punctuation
    # 3. remove stopwords to reduce noise
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)

def get_embedding(text: str):
    # Generate SBERT embedding (vector representation) for the processed text
    clean_text = preprocess(text)
    return sbert_model.encode(clean_text)

def compute_similarity(vec1, vec2):
    # Compute cosine similarity between two vectors; closer to 1 means stronger similarity
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def extract_skills_ner(text: str):
    # Extract potential technical skills using:
    # 1. spaCy NER for organizations, products, and tech-related entities
    # 2. regex to capture any tech-like alphanumeric terms (e.g., libraries, tools, versions)
    doc = nlp(text)
    skills = set()

    # Add spaCy-detected named entities
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "FAC"]:
            skills.add(ent.text.lower())

    # Use regex to detect any tech-like term with letters, numbers, +, #, -, or .
    tech_keywords = re.findall(r"\b[a-zA-Z0-9\+\#\-\._]{2,}\b", text)
    for kw in tech_keywords:
        kw = kw.lower()
        if kw not in STOPWORDS:
            skills.add(kw)

    return skills

def keyword_overlap(resume: str, jd: str) -> float:
    # Calculate the proportion of JD keywords present in the resume
    resume_skills = extract_skills_ner(resume)
    jd_skills = extract_skills_ner(jd)
    if not jd_skills:
        return 0.0
    return len(resume_skills & jd_skills) / len(jd_skills)

def compute_final_score(embedding_score: float, keyword_score: float, alpha: float = 0.5):
    # Combine semantic similarity and keyword overlap into a final weighted score
    return alpha * embedding_score + (1 - alpha) * keyword_score

def extract_text_from_pdf(file):
    # Extract text from a PDF using PyPDF2 (lightweight and Streamlit-friendly)
    try:
        text = ""
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
        return text.strip()
    except Exception as e:
        st.error(f"Could not read PDF: {e}. Please paste the text manually.")
        return ""

def generate_feedback(resume: str, jd: str, embedding_score: float, keyword_score: float):
    # Provide a human-readable explanation for the computed match score
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

# STREAMLIT USER INTERFACE AND ERROR HANDLING

# Upload areas for PDFs
col1, col2 = st.columns(2)
with col1:
    st.subheader("Upload Resume (PDF)")
    resume_file = st.file_uploader("Upload Resume", type=["pdf"], key="resume_upload")

with col2:
    st.subheader("Upload Job Description (PDF)")
    jd_file = st.file_uploader("Upload JD", type=["pdf"], key="jd_upload")

# Text input areas as a fallback if PDFs are not provided
st.subheader("OR Paste Text Instead")
resume_input = st.text_area("Paste Resume Text", height=180, disabled=bool(resume_file))
jd_input = st.text_area("Paste Job Description Text", height=180, disabled=bool(jd_file))

# Enforce single input method: stop execution if both PDF and text are provided
if (resume_file and resume_input.strip()) or (jd_file and jd_input.strip()):
    st.error("Please use either PDF upload OR text input for each field, not both.")
    st.stop()

# SCORE COMPUTATION AND DISPLAY

if st.button("Compute Match Score"):
    if not (resume_file or resume_input.strip()) or not (jd_file or jd_input.strip()):
        st.warning("Please provide both resume and job description.")
    else:
        # Show a progress bar for better UX during processing
        progress_bar = st.progress(0)
        for percent in range(0, 101, 10):
            time.sleep(0.05)  # Simulate gradual loading effect
            progress_bar.progress(percent)

        # Extract text from PDFs or fallback to manual text input
        resume_text = extract_text_from_pdf(resume_file) if resume_file else resume_input
        jd_text = extract_text_from_pdf(jd_file) if jd_file else jd_input

        # Start MLflow tracking if enabled; otherwise just compute directly
        context = mlflow.start_run() if MLFLOW_ENABLED else st.spinner("Computing match score...")
        with context:
            resume_vec = get_embedding(preprocess(resume_text))
            jd_vec = get_embedding(preprocess(jd_text))
            embedding_score = compute_similarity(resume_vec, jd_vec)
            overlap_score = keyword_overlap(resume_text, jd_text)
            final_score = compute_final_score(embedding_score, overlap_score)

            if MLFLOW_ENABLED:
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
                    f"<div style='color: lightgreen !important; font-weight: bold !important'>{highlighted}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.info("No overlapping technical keywords found. Add relevant tools or technologies.")

            st.markdown("#### Why You Got This Score:")
            feedback = generate_feedback(resume_text, jd_text, embedding_score, overlap_score)
            st.markdown(
                f"<div style='color: gold !important; background-color:#1e1e1e !important; "
                f"padding:10px; border-radius:5px'>{feedback}</div>",
                unsafe_allow_html=True
            )

            st.download_button("Download Feedback Report", feedback, file_name="resume_match_feedback.txt")