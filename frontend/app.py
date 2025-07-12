import streamlit as st
import requests

st.title("Resume ↔ Job Matcher")

resume = st.text_area("Paste your Resume:")
jd = st.text_area("Paste Job Description:")

if st.button("Compute Match Score"):
    if not resume or not jd:
        st.warning("Please paste both resume and job description.")
    else:
        try:
            response = requests.post("http://localhost:8000/match", json={"resume": resume, "jd": jd})
            response.raise_for_status()
            data = response.json()
            score = data["match_score"]
            overlap = data["overlap_keywords"]

            st.success(f"Match Score: **{score * 100:.2f}%**")

            st.markdown("#### Overlapping Keywords")
            if overlap:
                st.markdown(f"<span style='color:green; font-weight:bold'>{', '.join(overlap)}</span>", unsafe_allow_html=True)
            else:
                st.warning("No overlapping keywords found. Tailor your resume more specifically.")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
        except ValueError:
            st.error("Invalid response from backend.")