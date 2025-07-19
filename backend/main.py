import os
from fastapi import FastAPI
from pydantic import BaseModel
from model_utils import (
    preprocess,
    get_embedding,
    compute_similarity,
    keyword_overlap,
    compute_final_score,
    extract_skills_ner
)
import mlflow
from dotenv import load_dotenv

# ENVIRONMENT VARIABLES LOADING
# Loads environment variables from the .env file into the system environment.
# Ensures configuration (e.g., IS_LOCAL flag) is consistent across development and production.
load_dotenv()

# BACKEND INITIALIZATION
# FastAPI serves as the lightweight and high-performance web framework.
# All API endpoints and request/response handling are defined here.
app = FastAPI(title="ResumeFit Backend", version="1.2")

# ENVIRONMENT AND MLFLOW CONFIGURATION
# IS_LOCAL determines whether MLflow logging is enabled.
# It is read from an environment variable or the .env file for flexibility between local and deployed environments.
IS_LOCAL = os.getenv("IS_LOCAL", "false").lower() == "true"

# If running locally, attempt to connect to the MLflow tracking server.
# Any failure to connect gracefully disables experiment tracking.
if IS_LOCAL:
    try:
        mlflow.set_tracking_uri("http://127.0.0.1:6000")  # Local MLflow tracking server URI
        mlflow.set_experiment("ResumeFit Backend")        # Group all runs under this experiment
        MLFLOW_ENABLED = True
    except Exception:
        MLFLOW_ENABLED = False
else:
    MLFLOW_ENABLED = False


# REQUEST AND RESPONSE MODELS
# Pydantic models enforce strict typing for incoming requests and outgoing responses.

class MatchRequest(BaseModel):
    # Incoming request must include plain-text versions of the resume and job description.
    resume_text: str
    jd_text: str

class MatchResponse(BaseModel):
    # Outgoing response includes all relevant scoring components and matched keywords.
    embedding_score: float        # Semantic similarity score (0 to 1)
    keyword_score: float          # Direct keyword overlap score (0 to 1)
    final_score: float            # Weighted combination of embedding_score and keyword_score
    matched_keywords: list[str]   # List of technical skills/tools appearing in both resume and job description


# HEALTH CHECK ENDPOINT
# Confirms backend availability and whether MLflow logging is enabled (useful for debugging).
@app.get("/")
def root():
    return {
        "message": "ResumeFit Backend is running!",
        "mlflow_enabled": MLFLOW_ENABLED
    }


# MATCH COMPUTATION ENDPOINT
# Primary API endpoint:
# 1. Accepts raw text from a resume and job description.
# 2. Processes the text to compute semantic and keyword-based similarity.
# 3. Returns detailed scores and matched technical keywords for frontend visualization.
@app.post("/compute_match", response_model=MatchResponse)
def compute_match(data: MatchRequest):

    # STEP 1: GENERATE SEMANTIC EMBEDDINGS
    # Transforms resume and JD into high-dimensional vectors for semantic comparison.
    resume_vec = get_embedding(data.resume_text)
    jd_vec = get_embedding(data.jd_text)

    # STEP 2: CALCULATE SIMILARITY SCORES
    # Embedding-based semantic similarity (meaning-level match).
    # Keyword overlap based on shared technical skills or tools.
    embedding_score = compute_similarity(resume_vec, jd_vec)
    keyword_score = keyword_overlap(data.resume_text, data.jd_text)

    # STEP 3: COMPUTE FINAL WEIGHTED SCORE
    # Combines semantic similarity and keyword overlap into one interpretable metric.
    final_score = compute_final_score(embedding_score, keyword_score)

    # STEP 4: EXTRACT OVERLAPPING TECHNICAL KEYWORDS
    # Uses both NER and regex detection to identify shared tools, programming languages, and platforms.
    matched_keywords = sorted(
        list(extract_skills_ner(data.resume_text) & extract_skills_ner(data.jd_text))
    )

    # STEP 5: OPTIONAL MLFLOW LOGGING
    # Tracks parameters and metrics locally for experiment reproducibility and performance monitoring.
    if MLFLOW_ENABLED:
        with mlflow.start_run():
            mlflow.log_param("resume_length", len(data.resume_text))
            mlflow.log_param("jd_length", len(data.jd_text))
            mlflow.log_metric("embedding_score", embedding_score)
            mlflow.log_metric("keyword_score", keyword_score)
            mlflow.log_metric("final_score", final_score)
            mlflow.log_param("matched_keywords", ", ".join(matched_keywords))

    # STEP 6: RETURN RESULTS
    # Sends structured response back to the frontend for visualization.
    return MatchResponse(
        embedding_score=embedding_score,
        keyword_score=keyword_score,
        final_score=final_score,
        matched_keywords=matched_keywords
    )