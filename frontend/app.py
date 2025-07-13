import streamlit as st
import requests
import fitz  # PyMuPDF

# --- Page Config ---
st.set_page_config(
    page_title="ResumeFit Matcher",
    layout="centered"
)

st.markdown("""
    <style>
    .reportview-container {
        background: #f9f9f9;
        color: #333;
        font-family: "Segoe UI", sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5em 1em;
        border-radius: 8px;
    }
    .stTextArea textarea {
        background-color: #fffdf9;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Resume ↔ Job Matcher")
st.markdown("Upload or paste your **Resume** and **Job Description** below to check match compatibility.")

# Helper to Extract Text from PDF
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

# PDF Uploads
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Resume (PDF)")
    resume_file = st.file_uploader("Upload Resume", type=["pdf"], key="resume_upload")

with col2:
    st.subheader("Upload Job Description (PDF)")
    jd_file = st.file_uploader("Upload JD", type=["pdf"], key="jd_upload")

# Text Fallback
resume_text = ""
jd_text = ""

if resume_file:
    resume_text = extract_text_from_pdf(resume_file)

if jd_file:
    jd_text = extract_text_from_pdf(jd_file)

st.subheader("OR Paste Text Instead")

resume_input = st.text_area("Paste Resume Text", height=180)
jd_input = st.text_area("Paste Job Description Text", height=180)

if not resume_text:
    resume_text = resume_input
if not jd_text:
    jd_text = jd_input

# Match Score Button
if st.button("🔍 Compute Match Score"):
    if not resume_text or not jd_text:
        st.warning("Please upload or paste both Resume and Job Description.")
    else:
        try:
            response = requests.post("https://resumefit-semantic-matcher.onrender.com", json={"resume": resume_text, "jd": jd_text})
            response.raise_for_status()
            data = response.json()
            score = data["match_score"]
            overlap = data["overlap_keywords"]

            st.success(f"Match Score: **{score * 100:.2f}%**")

            st.markdown("#### Overlapping Keywords:")
            if overlap:
                highlighted = ", ".join([f"`{word}`" for word in sorted(overlap)])
                st.markdown(f"<div style='color: green; font-weight: bold'>{highlighted}</div>", unsafe_allow_html=True)
            else:
                st.info("No overlapping keywords found. Try tailoring your resume.")

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
        except ValueError:
            st.error("Invalid response from backend.")