from fastapi import FastAPI
from pydantic import BaseModel
from models.embedding_model import get_embedding
from models.matcher import compute_similarity
import mlflow

app = FastAPI()

class MatchRequest(BaseModel):
    resume: str
    jd: str

@app.post("/match")
def match_score(data: MatchRequest):
    with mlflow.start_run():
        resume_vec = get_embedding(data.resume)
        jd_vec = get_embedding(data.jd)
        score = compute_similarity(resume_vec, jd_vec)

        mlflow.log_param("model", "SBERT")
        mlflow.log_param("resume_length", len(data.resume))
        mlflow.log_param("jd_length", len(data.jd))
        mlflow.log_metric("match_score", float(score))  # fix here

        return {"match_score": float(round(score, 3))}  # fix here