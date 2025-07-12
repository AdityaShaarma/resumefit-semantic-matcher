import streamlit as st
import requests

st.title("🔍 Resume ↔ Job Matcher")

resume = st.text_area("Paste your Resume:")
jd = st.text_area("Paste Job Description:")

if st.button("Compute Match Score"):
    response = requests.post("http://localhost:8000/match", json={"resume": resume, "jd": jd})
    score = response.json()["match_score"]
    st.success(f"Match Score: {score * 100:.2f}%")