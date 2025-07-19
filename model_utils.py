import re
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords

# INITIAL SETUP
# download NLTK stopwords if not already available
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

# load SBERT model for semantic similarity; forced to CPU for better compatibility on Hugging Face
sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# curated set of technical keywords to boost keyword-based scoring
TECH_KEYWORDS = {
    "python", "java", "c++", "c#", "r", "sql", "nosql", "hadoop",
    "pandas", "numpy", "scikit-learn", "tensorflow", "keras",
    "pytorch", "xgboost", "transformers", "bert", "gpt",
    "azure", "aws", "gcp", "databricks", "powerbi", "tableau"
}

def preprocess(text: str) -> str:
    # cleans and normalizes text for semantic analysis
    # 1. lowercases everything
    # 2. removes punctuation and special characters
    # 3. removes stopwords to reduce noise
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

def get_embedding(text: str) -> np.ndarray:
    # converts text into a semantic vector using SBERT
    clean_text = preprocess(text)
    return sbert_model.encode(clean_text)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    # computes cosine similarity between two embeddings (0 to 1 range)
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def extract_keywords(text: str) -> set:
    # extracts meaningful technical keywords using regex + curated list
    tokens = set(re.findall(r"\b[a-zA-Z\+\#\d]+\b", text.lower()))
    filtered = {t for t in tokens if t in TECH_KEYWORDS}
    return filtered

def weighted_keyword_score(resume: str, jd: str) -> float:
    # computes weighted keyword overlap, giving higher weight to rare/specialized skills
    resume_skills = extract_keywords(resume)
    jd_skills = extract_keywords(jd)
    if not jd_skills:
        return 0.0
    weights = {skill: 2.0 if skill in {"pytorch", "xgboost", "transformers"} else 1.0 for skill in jd_skills}
    matched_weight = sum(weights[s] for s in resume_skills & jd_skills if s in weights)
    total_weight = sum(weights.values())
    return matched_weight / total_weight if total_weight > 0 else 0.0

def compute_final_score(resume: str, jd: str, alpha: float = 0.6) -> tuple:
    # blends semantic similarity and keyword overlap into a final score
    resume_vec = get_embedding(resume)
    jd_vec = get_embedding(jd)
    semantic_score = cosine_similarity(resume_vec, jd_vec)
    keyword_score = weighted_keyword_score(resume, jd)
    final_score = alpha * semantic_score + (1 - alpha) * keyword_score
    return final_score, semantic_score, keyword_score

def generate_feedback(resume: str, jd: str, semantic_score: float, keyword_score: float) -> str:
    # provides recruiter-friendly suggestions to improve resume alignment
    feedback = []
    missing_keywords = extract_keywords(jd) - extract_keywords(resume)

    if semantic_score < 0.5:
        feedback.append(f"Your resume is not contextually close to the job description (semantic similarity: {semantic_score:.2f}).")

    if keyword_score < 0.5 and missing_keywords:
        feedback.append(f"Highlight these important skills to stand out: {', '.join(sorted(missing_keywords))}.")

    if not feedback:
        feedback.append("Your resume strongly aligns with the job requirements. Well done!")

    return " ".join(feedback)