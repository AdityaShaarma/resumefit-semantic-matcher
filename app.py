import streamlit as st
import PyPDF2  # Library for reading and extracting text from PDF files
import re
import nltk
import mlflow
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import time  # Used to make progress bar updates visually smooth

# INITIAL SETUP: DOWNLOAD RESOURCES, LOAD MODELS, CONFIG

# Download stopwords on first run to prevent errors in new environments
# Stopwords (e.g., "the", "and", "of") are removed to reduce noise in text analysis
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# Load a compact SBERT (Sentence-BERT) model to create semantic embeddings
# "all-MiniLM-L6-v2" is lightweight and fast but provides high-quality semantic similarity
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load spaCy's English model for Named Entity Recognition (NER)
# If the pre-trained model is unavailable (e.g., restricted environment), fall back to a blank English model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = spacy.blank("en")

# Label used in MLflow tracking for model identification
MODEL_NAME = "SBERT"

# Attempt to connect to a local MLflow server to track experiments
# If no server is found, tracking will be disabled (does not break app functionality)
try:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("ResumeFit Matcher")
    MLFLOW_ENABLED = True
except Exception:
    MLFLOW_ENABLED = False

# STREAMLIT PAGE CONFIGURATION AND CUSTOM THEMING

# Set the page title and layout for better UX
st.set_page_config(page_title="ResumeFit Matcher", layout="centered")

# Apply custom CSS styling for a clean, dark-themed UI
st.markdown("""
    <style>
    .reportview-container {
        background: #0e1117 !important;  /* Dark background for modern look */
        color: #fafafa !important;       /* Light text for contrast */
        font-family: "Segoe UI", sans-serif !important;
    }
    .stButton>button {
        background-color: #4CAF50 !important;  /* Green action button */
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

# Page title and short instructions
st.title("Resume ↔ Job Matcher")
st.markdown("Upload or paste your **Resume** and **Job Description** below to check match compatibility.")

# TEXT PROCESSING AND NLP FUNCTIONS

def preprocess(text: str) -> str:
    # Standardize text for semantic analysis:
    # 1. Convert to lowercase to avoid treating "Python" and "python" differently
    # 2. Remove punctuation and special characters
    # 3. Remove stopwords to reduce noise and focus on meaningful terms
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)

def get_embedding(text: str):
    # Convert cleaned text into a numerical vector (embedding) using SBERT
    # Embeddings capture semantic meaning, enabling similarity comparisons
    clean_text = preprocess(text)
    return sbert_model.encode(clean_text)

def compute_similarity(vec1, vec2):
    # Compute cosine similarity between two vectors:
    # - Result ranges from 0 (no similarity) to 1 (identical meaning)
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def extract_skills_ner(text: str):
    # Identify potential technical skills or tools mentioned in the text
    # Combines two techniques:
    # 1. spaCy NER: Detects organizations, products, or tech-related named entities
    # 2. Regex: Captures tech-like terms (alphanumeric, +, #, -, .) that might not be standard entities
    doc = nlp(text)
    skills = set()

    # Named Entity Recognition (NER) extraction
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "FAC"]:
            skills.add(ent.text.lower())

    # Regex-based detection for libraries, languages, tools, and version numbers
    tech_keywords = re.findall(r"\b[a-zA-Z0-9\+\#\-\._]{2,}\b", text)
    for kw in tech_keywords:
        kw = kw.lower()
        if kw not in STOPWORDS:
            skills.add(kw)
    return skills

def keyword_overlap(resume: str, jd: str) -> float:
    # Calculate the percentage of job description keywords that appear in the resume
    resume_skills = extract_skills_ner(resume)
    jd_skills = extract_skills_ner(jd)
    if not jd_skills:
        return 0.0
    return len(resume_skills & jd_skills) / len(jd_skills)

def compute_final_score(embedding_score: float, keyword_score: float, alpha: float = 0.5):
    # Combine semantic similarity and keyword overlap into a final weighted score
    # alpha controls weighting:
    # - alpha closer to 1 → prioritize semantic similarity
    # - alpha closer to 0 → prioritize keyword overlap
    return alpha * embedding_score + (1 - alpha) * keyword_score

def extract_text_from_pdf(file, progress_bar=None, label="Processing PDF"):
    # Extract all text from a PDF file, updating a progress bar for real-time UX feedback
    # Steps:
    # 1. Read each page of the PDF sequentially
    # 2. Update progress bar based on pages processed
    # 3. Small delay (0.02s) for smoother animation (not needed for speed but improves UX)
    try:
        text = ""
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)

        if progress_bar:
            progress_bar.progress(0, text=label)

        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text() or ""
            text += page_text

            if progress_bar:
                percent = int(((i + 1) / total_pages) * 100)
                progress_bar.progress(percent, text=f"{label} ({percent}%)")
                time.sleep(0.02)

        if progress_bar:
            progress_bar.progress(100, text=f"{label} (Done)")
            time.sleep(0.2)
        return text.strip()
    except Exception as e:
        st.error(f"Could not read PDF: {e}. Please paste the text manually.")
        return ""

def generate_feedback(resume: str, jd: str, embedding_score: float, keyword_score: float):
    # Provide a clear explanation of the computed match score
    # Includes:
    # - Semantic similarity evaluation
    # - Missing technical skills compared to JD
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

# File uploaders for PDF inputs
col1, col2 = st.columns(2)
with col1:
    st.subheader("Upload Resume (PDF)")
    resume_file = st.file_uploader("Upload Resume", type=["pdf"], key="resume_upload")

with col2:
    st.subheader("Upload Job Description (PDF)")
    jd_file = st.file_uploader("Upload JD", type=["pdf"], key="jd_upload")

# Text fallback areas if PDFs are not provided
st.subheader("OR Paste Text Instead")
resume_input = st.text_area("Paste Resume Text", height=180, disabled=bool(resume_file))
jd_input = st.text_area("Paste Job Description Text", height=180, disabled=bool(jd_file))

# Prevent users from mixing input methods (PDF + text)
if (resume_file and resume_input.strip()) or (jd_file and jd_input.strip()):
    st.error("Please use either PDF upload OR text input for each field, not both.")
    st.stop()

# SCORE COMPUTATION AND OUTPUT DISPLAY

if st.button("Compute Match Score"):
    if not (resume_file or resume_input.strip()) or not (jd_file or jd_input.strip()):
        st.warning("Please provide both resume and job description.")
    else:
        # Show real-time progress for PDF text extraction
        resume_progress = st.progress(0)
        jd_progress = st.progress(0)

        resume_text = extract_text_from_pdf(resume_file, resume_progress, "Processing Resume") if resume_file else resume_input
        jd_text = extract_text_from_pdf(jd_file, jd_progress, "Processing Job Description") if jd_file else jd_input

        # Perform scoring inside a tracked MLflow run (if enabled)
        context = mlflow.start_run() if MLFLOW_ENABLED else st.spinner("Computing match score...")
        with context:
            # Generate semantic embeddings and compute similarity scores
            resume_vec = get_embedding(preprocess(resume_text))
            jd_vec = get_embedding(preprocess(jd_text))
            embedding_score = compute_similarity(resume_vec, jd_vec)
            overlap_score = keyword_overlap(resume_text, jd_text)
            final_score = compute_final_score(embedding_score, overlap_score)

            # Log parameters and metrics to MLflow for experiment tracking
            if MLFLOW_ENABLED:
                mlflow.log_param("model", MODEL_NAME)
                mlflow.log_param("resume_length", len(resume_text))
                mlflow.log_param("jd_length", len(jd_text))
                mlflow.log_metric("embedding_score", embedding_score)
                mlflow.log_metric("keyword_overlap", overlap_score)
                mlflow.log_metric("final_match_score", final_score)

            # Display final match score
            st.success(f"Match Score: {final_score * 100:.2f}%")

            # Show overlapping technical keywords (green highlighted tags)
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

            # Display reasoning behind the score
            st.markdown("#### Why You Got This Score:")
            feedback = generate_feedback(resume_text, jd_text, embedding_score, overlap_score)
            st.markdown(
                f"<div style='color: gold !important; background-color:#1e1e1e !important; "
                f"padding:10px; border-radius:5px'>{feedback}</div>",
                unsafe_allow_html=True
            )

            # Provide option to download the feedback as a text report
            st.download_button("Download Feedback Report", feedback, file_name="resume_match_feedback.txt")