import gradio as gr
import PyPDF2
import time
import tempfile
import traceback
from model_utils import compute_final_score, extract_keywords

# helper function to extract text from PDFs
# handles empty or malformed pages gracefully
def extract_text_from_pdf(pdf_file):
    if not pdf_file:
        return ""
    extracted_text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            extracted_text += page.extract_text() or ""
            time.sleep(0.02)  # adds small delay for smoother progress animation
    except Exception:
        return ""
    return extracted_text.strip()

# generates recruiter-friendly feedback without numerical scores
# focuses on actionable suggestions and overall fit
def recruiter_style_feedback(resume_text, jd_text, semantic_score, keyword_score):
    missing_keywords = extract_keywords(jd_text) - extract_keywords(resume_text)
    feedback_parts = []

    if semantic_score >= 0.7 and keyword_score >= 0.6:
        feedback_parts.append("Your resume is a strong fit and aligns well with the job description.")
    elif semantic_score >= 0.5:
        feedback_parts.append("Your resume is contextually relevant, but it could better highlight some technical skills.")
    else:
        feedback_parts.append("Your resume shows limited alignment with the job description. Consider tailoring it further.")

    if missing_keywords:
        feedback_parts.append(f"Consider adding or emphasizing these key skills: {', '.join(sorted(missing_keywords))}.")
    else:
        feedback_parts.append("Great job showcasing all the key technical skills required!")

    return " ".join(feedback_parts)

# helper to decide color for visual bars
# green for strong match, yellow for average, red for weak alignment
def get_bar_color(score):
    if score >= 0.7:
        return "#4CAF50"  # green
    elif score >= 0.4:
        return "#FFC107"  # yellow
    else:
        return "#F44336"  # red

# main matching function that handles computation and output preparation
def match_resume(resume_text, jd_text, resume_pdf, jd_pdf, progress=gr.Progress()):
    try:
        if resume_pdf:
            progress(0.1, desc="Extracting resume PDF...")
            resume_text = extract_text_from_pdf(resume_pdf)
        if jd_pdf:
            progress(0.2, desc="Extracting job description PDF...")
            jd_text = extract_text_from_pdf(jd_pdf)

        if not resume_text.strip() or not jd_text.strip():
            return (
                "Input Error",
                "Please provide both a resume and job description (paste text or upload PDFs).",
                0.0, 0.0, 0.0,
                "No keywords to display.",
                None,
                get_bar_color(0.0),
                get_bar_color(0.0),
                get_bar_color(0.0)
            )

        progress(0.5, desc="Analyzing semantic similarity and keyword relevance...")
        final_score, semantic_score, keyword_score = compute_final_score(resume_text, jd_text)

        progress(0.7, desc="Extracting matched technical keywords...")
        matched_keywords = extract_keywords(resume_text) & extract_keywords(jd_text)
        matched_keywords_text = (
            ", ".join(sorted(matched_keywords)) if matched_keywords else "No matched technical keywords found."
        )

        progress(1.0, desc="Generating actionable feedback...")
        feedback = recruiter_style_feedback(resume_text, jd_text, semantic_score, keyword_score)

        report_content = (
            f"Resume ↔ Job Match Report\n\n"
            f"Final Match Score: {final_score * 100:.2f}%\n"
            f"Matched Keywords:\n{matched_keywords_text}\n\n"
            f"Feedback & Suggestions:\n{feedback}\n"
        )
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
        tmp_file.write(report_content)
        tmp_file.close()

        return (
            f"{final_score * 100:.2f}%",
            feedback,
            semantic_score,
            keyword_score,
            final_score,
            matched_keywords_text,
            tmp_file.name,
            get_bar_color(semantic_score),
            get_bar_color(keyword_score),
            get_bar_color(final_score)
        )

    except Exception as e:
        error_trace = traceback.format_exc()
        return (
            "Processing Error",
            f"An unexpected error occurred:\n\n{error_trace}",
            0.0, 0.0, 0.0,
            "No keywords to display due to an error.",
            None,
            get_bar_color(0.0),
            get_bar_color(0.0),
            get_bar_color(0.0)
        )

# gradio interface with custom font and color-coded score bars
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="gray")) as demo:
    gr.HTML("""
    <style>
        * {
            font-family: Arial, Helvetica, sans-serif !important;
        }
    </style>
    """)

    gr.Markdown("# Resume ↔ Job Matcher")
    gr.Markdown("Upload or paste your **Resume** and **Job Description** to get a recruiter-style match evaluation.")

    with gr.Row():
        with gr.Column():
            resume_input = gr.Textbox(label="Paste Resume Text", lines=10, placeholder="Paste your resume text here...")
            resume_pdf = gr.File(label="Upload Resume (PDF)", file_types=[".pdf"])
        with gr.Column():
            jd_input = gr.Textbox(label="Paste Job Description Text", lines=10, placeholder="Paste the job description here...")
            jd_pdf = gr.File(label="Upload Job Description (PDF)", file_types=[".pdf"])

    submit_btn = gr.Button("Compute Match Score", variant="primary")

    final_score_label = gr.Label(label="Final Match Score")
    feedback_box = gr.Textbox(label="Feedback & Suggestions", interactive=False, lines=4)

    gr.Markdown("### Visual Breakdown of Scores")
    semantic_bar = gr.Slider(label="Semantic Similarity", minimum=0, maximum=1, step=0.01, interactive=False)
    keyword_bar = gr.Slider(label="Keyword Match", minimum=0, maximum=1, step=0.01, interactive=False)
    overall_bar = gr.Slider(label="Overall Match Score", minimum=0, maximum=1, step=0.01, interactive=False)

    matched_keywords_box = gr.Textbox(label="Matched Keywords", interactive=False, lines=2)
    gr.Markdown("&nbsp;")
    download_report = gr.File(label="Download Feedback Report")

    submit_btn.click(
        fn=match_resume,
        inputs=[resume_input, jd_input, resume_pdf, jd_pdf],
        outputs=[
            final_score_label,
            feedback_box,
            semantic_bar,
            keyword_bar,
            overall_bar,
            matched_keywords_box,
            download_report,
            semantic_bar.style,
            keyword_bar.style,
            overall_bar.style
        ]
    )

demo.launch()