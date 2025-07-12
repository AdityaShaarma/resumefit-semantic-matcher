from fastapi import FastAPI
from pydantic import BaseModel
from models.embedding_model import get_embedding, preprocess
from models.matcher import compute_similarity, keyword_overlap, compute_final_score
import mlflow

app = FastAPI()

class MatchRequest(BaseModel):
    resume: str
    jd: str

@app.post("/match")
def match_score(data: MatchRequest):
    with mlflow.start_run():
        # Preprocessing
        resume_clean = preprocess(data.resume)
        jd_clean = preprocess(data.jd)

        # Embeddings (SBERT only)
        resume_vec = get_embedding(resume_clean)
        jd_vec = get_embedding(jd_clean)

        # Compute scores
        embedding_score = float(compute_similarity(resume_vec, jd_vec))
        overlap_score = float(keyword_overlap(resume_clean, jd_clean))
        final_score = float(compute_final_score(embedding_score, overlap_score))

        # Log to MLflow
        mlflow.log_param("model", "SBERT")
        mlflow.log_param("resume_length", len(data.resume))
        mlflow.log_param("jd_length", len(data.jd))
        mlflow.log_metric("embedding_score", embedding_score)
        mlflow.log_metric("keyword_overlap", overlap_score)
        mlflow.log_metric("final_match_score", final_score)

        # Return match score and overlapping keywords
        return {
            "match_score": round(final_score, 4),
            "overlap_keywords": list(set(resume_clean.split()) & set(jd_clean.split()))
        }