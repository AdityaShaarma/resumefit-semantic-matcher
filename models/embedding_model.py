import os
import re
import nltk
from dotenv import load_dotenv

load_dotenv()
nltk.download('stopwords')
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

# Always use SBERT now
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)


def get_embedding(text: str):
    clean_text = preprocess(text)
    return sbert_model.encode(clean_text)