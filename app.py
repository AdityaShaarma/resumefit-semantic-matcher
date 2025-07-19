import gradio as gr
import PyPDF2
import time
from model_utils import compute_final_score, generate_feedback, extract_keywords

# EXTRACT TEXT FROM PDF WITH PROGRESS FEEDBACK
def extract_text_from_pdf(pdf_file):
    if not pdf_file:
        return ""
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
        time.sleep(0.05)  # simulate reading delay for smoother UX
    return text.strip()

# MAIN LOGIC TO MATCH RESUME AND JOB DESCRIPTION
def match_resume(resume_text, jd_text, resume_pdf, jd_pdf):
    # determine whether to use text or PDF inputs
    if resume_pdf:
        resume_text = extract_text_from_pdf(resume_pdf)
    if jd_pdf:
        jd_text = extract_text_from_pdf(jd_pdf)

    if not resume_text.strip() or not jd_text.strip():
        return "Please provide both resume and job description.", "", "", "", None

    final_score, semantic_score, keyword_score = compute_final_score(resume_text, jd_text)
    feedback = generate_feedback(resume_text, jd_text, semantic_score, keyword_score)
    matched_keywords = extract_keywords(resume_text) & extract_keywords(jd_text)

    # prepare a downloadable feedback report
    report_content = (
        f"Resume ↔ Job Match Report\n\n"
        f"Final Match Score: {final_score * 100:.2f}%\n"
        f"Semantic Similarity: {semantic_score:.2f}\n"
        f"Keyword Match: {keyword_score:.2f}\n\n"
        f"Matched Keywords: {', '.join(sorted(matched_keywords)) if matched_keywords else 'None'}\n\n"
        f"Feedback:\n{feedback}\n"
    )

    return (
        f"{final_score * 100:.2f}%",
        f"Semantic Similarity: {semantic_score:.2f} | Keyword Match: {keyword_score:.2f}",
        feedback,
        ", ".join(sorted(matched_keywords)) if matched_keywords else "No matched technical keywords found.",
        report_content
    )

# GRADIO INTERFACE DESIGN
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="gray")) as demo:
    gr.Markdown("# 📄 Resume ↔ Job Matcher")
    gr.Markdown("Upload or paste your **Resume** and **Job Description** to get an accurate match score with actionable feedback.")

    with gr.Row():
        with gr.Column():
            resume_input = gr.Textbox(label="Paste Resume Text", lines=10, placeholder="Paste your resume text here...")
            resume_pdf = gr.File(label="Upload Resume (PDF)", file_types=[".pdf"])
        with gr.Column():
            jd_input = gr.Textbox(label="Paste Job Description Text", lines=10, placeholder="Paste the job description here...")
            jd_pdf = gr.File(label="Upload Job Description (PDF)", file_types=[".pdf"])

    submit_btn = gr.Button("Compute Match Score", variant="primary")

    with gr.Row():
        output_score = gr.Label(label="Final Match Score")
        output_details = gr.Textbox(label="Detailed Analysis", interactive=False)

    feedback_box = gr.Textbox(label="Feedback & Suggestions", interactive=False, lines=4)
    matched_keywords_box = gr.Textbox(label="Matched Keywords", interactive=False)
    download_report = gr.File(label="Download Feedback Report")

    submit_btn.click(
        fn=match_resume,
        inputs=[resume_input, jd_input, resume_pdf, jd_pdf],
        outputs=[output_score, output_details, feedback_box, matched_keywords_box, download_report]
    )

    gr.Markdown("This tool uses semantic similarity + weighted keyword scoring for recruiter-level accuracy.")

demo.launch()