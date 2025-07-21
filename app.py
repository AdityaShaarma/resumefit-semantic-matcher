import gradio as gr
import PyPDF2
import time
import tempfile
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import mlflow
import hashlib
from mlflow import log_artifact, log_metric, log_param, set_experiment

from model_utils import compute_final_score, extract_keywords, generate_feedback

# MLflow INITIALIZATION

# Sset MLflow Tracking to my local folder
mlflow_folder = os.path.expanduser("~/Downloads/dev/ResumeFit Resume Matcher/mlruns")
os.makedirs(mlflow_folder, exist_ok=True)
mlflow.set_tracking_uri(f"file:{mlflow_folder}")

set_experiment("ResumeFit Matcher")

# extracts all text from a PDF using PyPDF2
# gracefully handles pages with missing or malformed text
def extract_text_from_pdf(pdf_file):
    if not pdf_file:
        return ""
    extracted_text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            extracted_text += page.extract_text() or ""
            time.sleep(0.02)  # adds a small delay for smooth progress bar animation in Gradio
    except Exception:
        return ""
    return extracted_text.strip()

# generates a colored horizontal bar for visualizing scores
# color changes based on score:
# green for high, yellow for medium, red for low
def colored_bar_html(score, label):
    color = "#4CAF50" if score >= 0.7 else "#FFC107" if score >= 0.4 else "#F44336"
    percentage = f"{score * 100:.1f}%"
    return f"""
    <div style="margin-bottom:8px">
        <strong>{label}: {percentage}</strong>
        <div style="background:#ddd;width:100%;height:15px;border-radius:5px;overflow:hidden">
            <div style="width:{score*100}%;height:15px;background:{color}"></div>
        </div>
    </div>
    """

# main function that calculates resume-to-job alignment
# returns match score, recruiter feedback, matched keywords, downloadable report, and visual bars
def match_resume(resume_text, jd_text, resume_pdf, jd_pdf, progress=gr.Progress()):
    try:
        # extract text from PDFs if provided by the user
        if resume_pdf:
            progress(0.1, desc="Extracting resume PDF...")
            resume_text = extract_text_from_pdf(resume_pdf)
        if jd_pdf:
            progress(0.2, desc="Extracting job description PDF...")
            jd_text = extract_text_from_pdf(jd_pdf)

        # validation to ensure both resume and job description have text
        if not resume_text.strip() or not jd_text.strip():
            return (
                "Input Error",
                "Please provide both a resume and job description.",
                "No matched keywords to display.",
                None,
                ""
            )

        # compute semantic similarity and keyword overlap scores
        progress(0.5, desc="Analyzing similarity...")
        final_score, semantic_score, keyword_score = compute_final_score(resume_text, jd_text)

        # extract overlapping technical keywords
        progress(0.7, desc="Extracting matched keywords...")
        matched_keywords = extract_keywords(resume_text) & extract_keywords(jd_text)
        matched_keywords_text = ", ".join(sorted(matched_keywords)) if matched_keywords else "No matched technical keywords found."

        # generate recruiter-style feedback
        progress(1.0, desc="Generating feedback...")
        feedback = generate_feedback(resume_text, jd_text, semantic_score, keyword_score)

        # prepare a downloadable report with scores, keywords, and feedback
        report_content = (
            f"Resume ↔ Job Match Report\n\n"
            f"Final Match Score: {final_score * 100:.2f}%\n"
            f"Matched Keywords:\n{matched_keywords_text}\n\n"
            f"Feedback:\n{feedback}"
        )
        timestamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        file_name = f"ResumeFit_Report_{timestamp}.txt"

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
        tmp_file.write(report_content)
        tmp_file.close()

        new_path = os.path.join(tempfile.gettempdir(), file_name)
        os.rename(tmp_file.name, new_path)

        # MLflow Logging
        with mlflow.start_run(run_name=f"Match_{timestamp}"):
            # Parameters (metadata)
            log_param("resume_text_hash", hashlib.md5(resume_text.encode()).hexdigest())
            log_param("jd_text_hash", hashlib.md5(jd_text.encode()).hexdigest())
            log_param("resume_length", len(resume_text.split()))
            log_param("jd_length", len(jd_text.split()))
            log_param("num_matched_keywords", len(matched_keywords))

            # Metrics (for analysis)
            log_metric("final_score", final_score)
            log_metric("semantic_score", semantic_score)
            log_metric("keyword_score", keyword_score)

            # Artifacts (save full report)
            log_artifact(new_path)

            # Tags (extra info for filtering later)
            mlflow.set_tag("matched_keywords", matched_keywords_text)
            mlflow.set_tag("timestamp", timestamp)

        # generate visual bars for semantic similarity, keyword match, and overall score
        visual_bars = (
            colored_bar_html(semantic_score, "Semantic Similarity") +
            colored_bar_html(keyword_score, "Keyword Match") +
            colored_bar_html(final_score, "Overall Match Score")
        )

        return f"{final_score * 100:.2f}%", feedback, matched_keywords_text, new_path, visual_bars

    except Exception:
        # handle unexpected errors gracefully and display traceback for debugging
        error_trace = traceback.format_exc()
        return (
            "Processing Error",
            f"An unexpected error occurred:\n\n{error_trace}",
            "No keywords due to error.",
            None,
            ""
        )

# build Gradio interface
# uses a clean layout with textboxes, PDF upload options, and downloadable report
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="gray")) as demo:
    gr.HTML("<style>* { font-family: Arial, Helvetica, sans-serif !important; }</style>")
    gr.Markdown("# Resume ↔ Job Matcher")
    gr.Markdown(
        "Upload or paste your **Resume** and **Job Description** to get recruiter-style feedback.\n"
        "Includes **semantic alignment scores**, **matched technical keywords**, and **overall match percentage**."
    )

    with gr.Row():
        with gr.Column():
            resume_input = gr.Textbox(label="Paste Resume Text", lines=10, placeholder="Paste your resume text...")
            resume_pdf = gr.File(label="Upload Resume (PDF)", file_types=[".pdf"])
        with gr.Column():
            jd_input = gr.Textbox(label="Paste Job Description Text", lines=10, placeholder="Paste the job description...")
            jd_pdf = gr.File(label="Upload Job Description (PDF)", file_types=[".pdf"])

    submit_btn = gr.Button("Compute Match Score", variant="primary")
    final_score_label = gr.Label(label="Final Match Score")
    feedback_box = gr.Textbox(label="Feedback & Suggestions", interactive=False, lines=4)
    matched_keywords_box = gr.Textbox(label="Matched Keywords", interactive=False, lines=2)
    visual_bars_html = gr.HTML()
    download_report = gr.File(label="Download Feedback Report")

    submit_btn.click(
        fn=match_resume,
        inputs=[resume_input, jd_input, resume_pdf, jd_pdf],
        outputs=[final_score_label, feedback_box, matched_keywords_box, download_report, visual_bars_html]
    )

demo.launch()