import numpy as np

def compute_similarity(vec1, vec2):
    """Cosine similarity between two vectors."""
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def keyword_overlap(resume: str, jd: str) -> float:
    """Keyword overlap score (Jaccard-like) between preprocessed texts."""
    resume_tokens = set(resume.split())
    jd_tokens = set(jd.split())
    if not jd_tokens:
        return 0.0
    return len(resume_tokens & jd_tokens) / len(jd_tokens)

def compute_final_score(embedding_score: float, keyword_score: float, alpha: float = 0.7) -> float:
    """Weighted average of embedding similarity and keyword overlap."""
    return alpha * embedding_score + (1 - alpha) * keyword_score