import re
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords

# download stopwords if not already available
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

# initialize SBERT for semantic similarity - forced to CPU for lightweight inference
sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# curated technical keywords list for stronger keyword extraction
# can be expanded further based on domains
TECH_KEYWORDS = {
    "python", "java", "c++", "c#", "r", "sql", "nosql",
    "pandas", "numpy", "scikit-learn", "tensorflow", "keras",
    "pytorch", "xgboost", "transformers", "bert", "gpt",
    "azure", "aws", "gcp", "databricks", "powerbi", "tableau"
}

def preprocess(text: str) -> str:
    # standardizes and cleans text before further processing
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

def get_embedding(text: str) -> np.ndarray:
    # generates high-dimensional semantic vector representation of text
    clean_text = preprocess(text)
    return sbert_model.encode(clean_text)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    # calculates cosine similarity between two embedding vectors
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def extract_keywords(text: str) -> set:
    # extracts relevant keywords from the text using:
    # 1. regex to find alphanumeric technical terms
    # 2. matching against curated tech keywords
    tokens = set(re.findall(r"\b[a-zA-Z\+\#\d]+\b", text.lower()))
    filtered = {t for t in tokens if t in TECH_KEYWORDS}
    return filtered

def keyword_overlap(resume: str, jd: str) -> float:
    # calculates proportion of jd technical keywords present in the resume
    resume_skills = extract_keywords(resume)
    jd_skills = extract_keywords(jd)
    if not jd_skills:
        return 0.0
    return len(resume_skills & jd_skills) / len(jd_skills)

def weighted_keyword_score(resume: str, jd: str) -> float:
    # assigns higher weight to rare or specialized skills
    resume_skills = extract_keywords(resume)
    jd_skills = extract_keywords(jd)
    if not jd_skills:
        return 0.0
    weights = {skill: 2.0 if skill in {"xgboost", "pytorch", "transformers", "gcp"} else 1.0 for skill in jd_skills}
    matched_weight = sum(weights[s] for s in resume_skills & jd_skills if s in weights)
    total_weight = sum(weights.values())
    return matched_weight / total_weight if total_weight > 0 else 0.0

def compute_final_score(resume: str, jd: str, alpha: float = 0.6) -> tuple:
    # combines semantic similarity and weighted keyword overlap into final score
    resume_vec = get_embedding(resume)
    jd_vec = get_embedding(jd)
    semantic_score = cosine_similarity(resume_vec, jd_vec)
    keyword_score = weighted_keyword_score(resume, jd)
    final_score = alpha * semantic_score + (1 - alpha) * keyword_score
    return final_score, semantic_score, keyword_score

def generate_feedback(resume: str, jd: str, semantic_score: float, keyword_score: float) -> str:
    # generates human-readable feedback to help improve resume alignment
    feedback_parts = []
    missing_keywords = extract_keywords(jd) - extract_keywords(resume)

    if semantic_score < 0.5:
        feedback_parts.append(f"Your resume does not closely match the job context (semantic similarity: {semantic_score:.2f}).")

    if keyword_score < 0.5 and missing_keywords:
        feedback_parts.append(f"Consider adding or highlighting these important skills: {', '.join(sorted(missing_keywords))}.")

    if not feedback_parts:
        feedback_parts.append("Your resume strongly aligns with the job requirements. Great work!")

    return " ".join(feedback_parts)