import streamlit as st
import requests

st.title("Resume ↔ Job Matcher")

resume = st.text_area("Paste your Resume:")
jd = st.text_area("Paste Job Description:")

if st.button("Compute Match Score"):
    try:
        response = requests.post("http://localhost:8000/match", json={"resume": resume, "jd": jd})
        response.raise_for_status()
        score = response.json()["match_score"]
        st.success(f"💡 Match Score: {score * 100:.2f}%")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
    except ValueError:
        st.error("Invalid response from backend (likely a crash or error). Check terminal 1 for logs.")