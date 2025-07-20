import gradio as gr
import PyPDF2
import time
import tempfile
import traceback
from model_utils import compute_final_score, extract_keywords, generate_bullet_point_feedback
from datetime import datetime
from zoneinfo import ZoneInfo 
import os

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
            time.sleep(0.02)  # small delay for smooth progress bar animation
    except Exception:
        return ""
    return extracted_text.strip()

# recruiter-style resume feedback
# focuses on alignment with job description instead of raw numbers
def recruiter_style_feedback(resume_text, jd_text, semantic_score, keyword_score):
    missing_keywords = extract_keywords(jd_text) - extract_keywords(resume_text)
    feedback_parts = []

    if semantic_score >= 0.7 and keyword_score >= 0.6:
        feedback_parts.append("Your resume strongly aligns with the job description. Great job!")
    elif semantic_score >= 0.5:
        feedback_parts.append("Your resume is contextually relevant but could better highlight technical expertise.")
    else:
        feedback_parts.append("Your resume shows limited alignment. Consider tailoring it for this role.")

    if missing_keywords:
        feedback_parts.append(f"Add or emphasize these skills to improve alignment: {', '.join(sorted(missing_keywords))}.")
    else:
        feedback_parts.append("You’ve covered all critical technical skills well.")

    return " ".join(feedback_parts)

# generates a horizontal color bar for visualizing scores
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

# formats bullet point feedback into a clean, visual HTML list
def format_bullet_feedback_html(feedback_list):
    if not feedback_list:
        return "<p style='color:green;font-weight:bold'>All bullet points look strong ✔</p>"

    html_output = "<ul style='padding-left:20px'>"
    for fb in feedback_list:
        # suggestions are marked in orange for easy spotting
        html_output += f"<li style='color:#FF9800;margin-bottom:5px'>{fb}</li>"
    html_output += "</ul>"
    return html_output

# main function that runs the matching and feedback logic
def match_resume(resume_text, jd_text, resume_pdf, jd_pdf, progress=gr.Progress()):
    try:
        # handle PDF extraction first if uploaded
        if resume_pdf:
            progress(0.1, desc="Extracting resume PDF...")
            resume_text = extract_text_from_pdf(resume_pdf)
        if jd_pdf:
            progress(0.2, desc="Extracting job description PDF...")
            jd_text = extract_text_from_pdf(jd_pdf)

        # basic validation check
        if not resume_text.strip() or not jd_text.strip():
            return (
                "Input Error",
                "Please provide both a resume and job description.",
                "No matched keywords to display.",
                None,
                "",
                "<p style='color:red'>No bullet point suggestions due to missing input.</p>"
            )

        # compute match scores
        progress(0.5, desc="Analyzing semantic similarity and keyword relevance...")
        final_score, semantic_score, keyword_score = compute_final_score(resume_text, jd_text)

        # extract overlapping technical keywords
        progress(0.7, desc="Extracting matched keywords...")
        matched_keywords = extract_keywords(resume_text) & extract_keywords(jd_text)
        matched_keywords_text = (
            ", ".join(sorted(matched_keywords)) if matched_keywords else "No matched technical keywords found."
        )

        # generate bullet point suggestions (only for weak ones)
        progress(0.9, desc="Analyzing resume bullet points...")
        bullet_feedback_list = generate_bullet_point_feedback(resume_text)
        bullet_feedback_html = format_bullet_feedback_html(bullet_feedback_list)

        # recruiter-style feedback for overall alignment
        progress(1.0, desc="Generating recruiter-style feedback...")
        feedback = recruiter_style_feedback(resume_text, jd_text, semantic_score, keyword_score)

        # write downloadable report to temp file
        report_content = (
            f"Resume ↔ Job Match Report\n\n"
            f"Final Match Score: {final_score * 100:.2f}%\n"
            f"Matched Keywords:\n{matched_keywords_text}\n\n"
            f"Feedback & Suggestions:\n{feedback}\n\n"
            f"Bullet Point Suggestions:\n"
            + ("\n".join(bullet_feedback_list) if bullet_feedback_list else "All bullet points look strong ✔")
        )

        timestamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        file_name = f"ResumeFit_Report_{timestamp}.txt"

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
        tmp_file.write(report_content)
        tmp_file.close()

        new_path = os.path.join(tempfile.gettempdir(), file_name)
        os.rename(tmp_file.name, new_path)

        # generate HTML visual score bars
        visual_bars = (
            colored_bar_html(semantic_score, "Semantic Similarity") +
            colored_bar_html(keyword_score, "Keyword Match") +
            colored_bar_html(final_score, "Overall Match Score")
        )

        return (
            f"{final_score * 100:.2f}%",
            feedback,
            matched_keywords_text,
            new_path,
            visual_bars,
            bullet_feedback_html
        )

    except Exception:
        error_trace = traceback.format_exc()
        return (
            "Processing Error",
            f"An unexpected error occurred:\n\n{error_trace}",
            "No keywords due to error.",
            None,
            "",
            "<p style='color:red'>No bullet point suggestions due to an error.</p>"
        )

# define interface layout and design
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="gray")) as demo:
    # set a consistent font for better readability
    gr.HTML("""
    <style>
        * { font-family: Arial, Helvetica, sans-serif !important; }
    </style>
    """)

    # title and instructions
    gr.Markdown("# Resume ↔ Job Matcher")
    gr.Markdown(
        "Upload or paste your **Resume** and **Job Description** to get recruiter-style feedback.\n"
        "Includes **semantic alignment scores** and **specific bullet point suggestions**."
    )

    # input section (resume and JD text or PDF)
    with gr.Row():
        with gr.Column():
            resume_input = gr.Textbox(label="Paste Resume Text", lines=10, placeholder="Paste your resume text...")
            resume_pdf = gr.File(label="Upload Resume (PDF)", file_types=[".pdf"])
        with gr.Column():
            jd_input = gr.Textbox(label="Paste Job Description Text", lines=10, placeholder="Paste the job description...")
            jd_pdf = gr.File(label="Upload Job Description (PDF)", file_types=[".pdf"])

    # action button
    submit_btn = gr.Button("Compute Match Score", variant="primary")

    # output section (score, recruiter feedback, keywords, visual bars, bullet suggestions)
    final_score_label = gr.Label(label="Final Match Score")
    feedback_box = gr.Textbox(label="Feedback & Suggestions", interactive=False, lines=4)
    matched_keywords_box = gr.Textbox(label="Matched Keywords", interactive=False, lines=2)
    visual_bars_html = gr.HTML()
    bullet_feedback_html = gr.HTML()
    download_report = gr.File(label="Download Feedback Report")

    # connect computation to button click
    submit_btn.click(
        fn=match_resume,
        inputs=[resume_input, jd_input, resume_pdf, jd_pdf],
        outputs=[
            final_score_label,
            feedback_box,
            matched_keywords_box,
            download_report,
            visual_bars_html,
            bullet_feedback_html
        ]
    )

# launch the app
demo.launch()