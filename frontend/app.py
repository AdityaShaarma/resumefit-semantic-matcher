import streamlit as st
import requests
import PyPDF2
import time

# PAGE CONFIGURATION AND CUSTOM STYLING
# Sets the title, layout, and dark-themed custom UI for better aesthetics
st.set_page_config(page_title="ResumeFit Matcher", layout="centered")

st.markdown("""
    <style>
    .reportview-container {
        background: #0e1117 !important;
        color: #fafafa !important;
        font-family: "Segoe UI", sans-serif !important;
    }
    .stButton>button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-weight: bold !important;
        padding: 0.5em 1em !important;
        border-radius: 8px !important;
    }
    .stTextArea textarea {
        background-color: #1e1e1e !important;
        color: white !important;
        border-radius: 8px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Resume ↔ Job Matcher")
st.markdown("Upload or paste your **Resume** and **Job Description** to check compatibility.")

# BACKEND ENDPOINT CONFIGURATION
# Update the URL when deploying to a public server
BACKEND_URL = "http://127.0.0.1:9000"

# PDF TEXT EXTRACTION FUNCTION WITH PROGRESS BAR
def extract_text_from_pdf(file):
    """
    Extracts text from a PDF file page by page while updating a progress bar.
    Provides real-time feedback for better user experience on large files.
    """
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)
        progress = st.progress(0)
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text() or ""
            text += page_text
            progress.progress((i + 1) / total_pages)
            time.sleep(0.03)  # Slight delay for smoother animation
        return text.strip()
    except Exception as e:
        st.error(f"Could not read PDF: {e}. Please paste the text manually.")
        return ""

# INPUT SECTIONS
col1, col2 = st.columns(2)
with col1:
    st.subheader("Upload Resume (PDF)")
    resume_file = st.file_uploader("Upload Resume", type=["pdf"], key="resume_upload")

with col2:
    st.subheader("Upload Job Description (PDF)")
    jd_file = st.file_uploader("Upload JD", type=["pdf"], key="jd_upload")

st.subheader("OR Paste Text Instead")
resume_input = st.text_area("Paste Resume Text", height=180, disabled=bool(resume_file))
jd_input = st.text_area("Paste Job Description Text", height=180, disabled=bool(jd_file))

# ENFORCE SINGLE INPUT METHOD
if (resume_file and resume_input.strip()) or (jd_file and jd_input.strip()):
    st.error("Please use either PDF upload OR text input for each field, not both.")
    st.stop()

# EXTRACT TEXT FROM PDFs OR USE PASTED TEXT
resume_text = extract_text_from_pdf(resume_file) if resume_file else resume_input
jd_text = extract_text_from_pdf(jd_file) if jd_file else jd_input

# MATCH SCORE COMPUTATION AND DISPLAY
if st.button("Compute Match Score"):
    if not resume_text or not jd_text:
        st.warning("Please provide both resume and job description.")
    else:
        with st.spinner("Computing match score..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/compute_match",
                    json={"resume_text": resume_text, "jd_text": jd_text}
                )

                if response.status_code == 200:
                    data = response.json()
                    final_score = data["final_score"]
                    embedding_score = data["embedding_score"]
                    keyword_score = data["keyword_score"]
                    matched_keywords = data.get("matched_keywords", [])

                    # COLOR-CODED FINAL MATCH SCORE
                    if final_score >= 0.75:
                        score_color = "#4CAF50"  # Green for strong match
                    elif 0.5 <= final_score < 0.75:
                        score_color = "#FF9800"  # Orange for moderate match
                    else:
                        score_color = "#F44336"  # Red for weak match

                    st.markdown(
                        f"<div style='background-color:{score_color}; color:white; "
                        f"padding:12px; border-radius:6px; font-size:18px; font-weight:bold;'>"
                        f"Final Match Score: {final_score * 100:.2f}%</div>",
                        unsafe_allow_html=True
                    )

                    # WHY YOU GOT THIS SCORE
                    st.markdown("### Why You Got This Score")

                    if final_score >= 0.75:
                        feedback_text = (
                            "Your resume strongly aligns with the job requirements. "
                            "You highlight relevant technical skills and experience effectively."
                        )
                    elif 0.5 <= final_score < 0.75:
                        feedback_text = (
                            "Your resume is a decent match but could be improved. "
                            "Consider adding missing technical skills and aligning descriptions more closely."
                        )
                    else:
                        feedback_text = (
                            "Your resume has low alignment with the job description. "
                            "Focus on emphasizing relevant technical expertise and rephrasing experiences to match the job requirements."
                        )

                    st.markdown(
                        f"<div style='color: gold; background-color:#1e1e1e; "
                        f"padding:12px; border-radius:6px; margin-bottom:20px;'>"
                        f"{feedback_text}</div>",
                        unsafe_allow_html=True
                    )

                    # MATCHED KEYWORDS DISPLAY AS TAGS
                    st.markdown("### Matched Keywords")
                    if matched_keywords:
                        keywords_html = " ".join(
                            [f"<span style='background-color:#4CAF50; color:white; "
                             f"padding:4px 8px; border-radius:5px; margin:2px; display:inline-block;'>"
                             f"{kw}</span>" for kw in matched_keywords]
                        )
                        st.markdown(keywords_html, unsafe_allow_html=True)
                    else:
                        st.info("No technical keywords matched between your resume and the job description.")

                    # ADD SPACING BEFORE DOWNLOAD BUTTON
                    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)

                    # DOWNLOADABLE FEEDBACK REPORT
                    st.download_button(
                        "Download Feedback Report",
                        f"Final Score: {final_score * 100:.2f}%\n\n{feedback_text}",
                        file_name="resume_match_feedback.txt"
                    )

                else:
                    st.error("Error: Could not compute match score. Please try again.")
            except Exception as e:
                st.error(f"An error occurred while contacting the backend: {e}")