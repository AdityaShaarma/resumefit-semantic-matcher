import os
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv

load_dotenv()
MODEL_TYPE = os.getenv("EMBEDDING_MODEL", "sbert").lower()

if MODEL_TYPE == "sbert":
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
else:
    openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text: str):
    if MODEL_TYPE == "sbert":
        return sbert_model.encode(text)
    elif MODEL_TYPE == "openai":
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response["data"][0]["embedding"]
    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")