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

        # Keyword overlap logic
        resume_words = set(resume.lower().split())
        jd_words = set(jd.lower().split())
        overlap = resume_words & jd_words

        st.success(f"Match Score: {score * 100:.2f}%")
        st.markdown("#### 🧠 Overlapping Keywords")
        if overlap:
            st.markdown(f"<span style='color:green; font-weight:bold'>{', '.join(overlap)}</span>", unsafe_allow_html=True)
        else:
            st.warning("No overlapping words found. Try tailoring your resume to the JD.")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
    except ValueError:
        st.error("Invalid response from backend.")