import re
import numpy as np
import nltk
import spacy
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from functools import lru_cache

# INITIAL SETUP
# Download and cache NLTK stopwords if not already present
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

# MODEL INITIALIZATION
# SBERT model is lazy-loaded and cached to reduce startup time on Hugging Face
@lru_cache(maxsize=1)
def get_sbert_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Load spaCy's English model for NER
# Fallback to a blank pipeline if unavailable to avoid runtime crashes
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = spacy.blank("en")

# TEXT PREPROCESSING
def preprocess(text: str) -> str:
    # Standardizes and cleans text before generating embeddings
    # Steps:
    # 1. Lowercase conversion for case-insensitive comparisons
    # 2. Remove punctuation and special characters
    # 3. Tokenization and stopword removal to reduce noise
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = [t for t in text.split() if t not in STOPWORDS]
    return " ".join(tokens)

# EMBEDDING GENERATION
def get_embedding(text: str) -> np.ndarray:
    # Converts preprocessed text into a high-dimensional semantic vector using SBERT
    model = get_sbert_model()
    return model.encode(preprocess(text))

# COSINE SIMILARITY
def compute_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    # Computes cosine similarity between two vectors
    # Value close to 1 indicates strong semantic similarity
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

# TECHNICAL KEYWORD EXTRACTION
def extract_skills_ner(text: str) -> set:
    # Extracts technical skills or relevant keywords from the text using:
    # 1. spaCy Named Entity Recognition (technologies, organizations, products)
    # 2. Regex-based keyword detection for programming languages, libraries, and cloud tools
    doc = nlp(text)
    skills = set()

    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "FAC"]:
            skills.add(ent.text.lower())

    tech_keywords = re.findall(r"\b[a-zA-Z\+\#\d]+\b", text)
    for kw in tech_keywords:
        kw = kw.lower()
        if kw not in STOPWORDS and len(kw) > 1:
            if (
                any(c.isdigit() for c in kw)
                or kw in [
                    "python", "java", "c++", "c#", "sql", "pandas", "numpy",
                    "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost",
                    "azure", "aws", "databricks", "bert", "gpt", "transformers"
                ]
            ):
                skills.add(kw)
    return skills

# KEYWORD OVERLAP SCORE
def keyword_overlap(resume: str, jd: str) -> float:
    # Calculates proportion of job description keywords present in the resume
    resume_skills = extract_skills_ner(resume)
    jd_skills = extract_skills_ner(jd)
    if not jd_skills:
        return 0.0
    return len(resume_skills & jd_skills) / len(jd_skills)

# FINAL MATCH SCORE
def compute_final_score(embedding_score: float, keyword_score: float, alpha: float = 0.5) -> float:
    # Combines semantic similarity and keyword overlap using weighted average
    return alpha * embedding_score + (1 - alpha) * keyword_score

# FEEDBACK GENERATION
def generate_feedback(resume: str, jd: str, embedding_score: float, keyword_score: float) -> list:
    # Creates short, recruiter-friendly feedback points explaining the score
    resume_skills = extract_skills_ner(resume)
    jd_skills = extract_skills_ner(jd)
    missing = jd_skills - resume_skills

    feedback = []
    if embedding_score < 0.5:
        feedback.append(f"Low semantic alignment detected (Embedding Score: {embedding_score:.2f}).")
    if keyword_score < 0.5:
        if missing:
            feedback.append(f"Missing important skills: {', '.join(missing)}.")
        else:
            feedback.append("Few technical keywords matched; align terminology with job description.")
    if not feedback:
        feedback.append("Your resume aligns strongly with the job description. Great work!")
    return feedback