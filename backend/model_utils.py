import re
import numpy as np
import nltk
import spacy
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords

# INITIAL SETUP
# Download NLTK stopwords if not already present
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

# MODEL INITIALIZATION
# SBERT model for semantic similarity, forced to CPU for compatibility across most servers
sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Attempt to load spaCy's small English model for NER
# If unavailable, fallback to a blank English pipeline (only tokenization)
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = spacy.blank("en")


# TEXT PREPROCESSING
def preprocess(text: str) -> str:
    # Standardizes and cleans text before generating embeddings
    # Steps:
    # 1. Lowercase conversion for case-insensitive comparisons
    # 2. Removal of punctuation and non-alphanumeric characters
    # 3. Tokenization into words
    # 4. Stopword removal to reduce noise
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)


# EMBEDDING GENERATION
def get_embedding(text: str) -> np.ndarray:
    # Converts cleaned text into a high-dimensional semantic vector using SBERT
    clean_text = preprocess(text)
    return sbert_model.encode(clean_text)


# COSINE SIMILARITY
def compute_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    # Computes cosine similarity between two embedding vectors
    # Returns a value between 0 and 1, where closer to 1 means higher semantic similarity
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


# TECHNICAL KEYWORD EXTRACTION
def extract_skills_ner(text: str) -> set:
    # Identifies potential technical skills and tools using:
    # 1. spaCy Named Entity Recognition (products, organizations, tech terms)
    # 2. Regex keyword detection for programming languages, libraries, and cloud tools
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
    # Useful for measuring direct technical skill alignment
    resume_skills = extract_skills_ner(resume)
    jd_skills = extract_skills_ner(jd)
    if not jd_skills:
        return 0.0
    return len(resume_skills & jd_skills) / len(jd_skills)


# FINAL MATCH SCORE
def compute_final_score(embedding_score: float, keyword_score: float, alpha: float = 0.5) -> float:
    # Combines semantic similarity (embedding_score) and keyword overlap into a final weighted score
    # alpha = weighting factor (0.5 = equal weight for semantic similarity and keyword overlap)
    return alpha * embedding_score + (1 - alpha) * keyword_score
