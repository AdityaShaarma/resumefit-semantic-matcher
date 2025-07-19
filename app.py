import gradio as gr
import PyPDF2
import time
from model_utils import compute_final_score, generate_feedback, extract_keywords

# PDF EXTRACTION WITH SIMULATED PROGRESS
def extract_text_from_pdf(pdf_file):
    if not pdf_file:
        return ""
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
        time.sleep(0.03)  # small delay to simulate reading progress smoothly
    return text.strip()

# MAIN MATCH COMPUTATION WITH PROGRESS FEEDBACK
def match_resume(resume_text, jd_text, resume_pdf, jd_pdf, progress=gr.Progress()):
    # prioritize PDFs over manual text if uploaded
    if resume_pdf:
        progress(0.1, desc="Extracting resume PDF...")
        resume_text = extract_text_from_pdf(resume_pdf)
    if jd_pdf:
        progress(0.2, desc="Extracting job description PDF...")
        jd_text = extract_text_from_pdf(jd_pdf)

    if not resume_text.strip() or not jd_text.strip():
        return "Please provide both resume and job description.", "", 0, 0, 0, "", None

    # step 1 – semantic and keyword computation
    progress(0.4, desc="Computing semantic similarity...")
    final_score, semantic_score, keyword_score = compute_final_score(resume_text, jd_text)

    # step 2 – keyword extraction
    progress(0.7, desc="Extracting matched keywords...")
    matched_keywords = extract_keywords(resume_text) & extract_keywords(jd_text)

    # step 3 – feedback generation
    progress(1.0, desc="Generating actionable feedback...")
    feedback = generate_feedback(resume_text, jd_text, semantic_score, keyword_score)

    # prepare downloadable report
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
        feedback,
        semantic_score,
        keyword_score,
        final_score,
        ", ".join(sorted(matched_keywords)) if matched_keywords else "No matched technical keywords found.",
        report_content
    )

# GRADIO INTERFACE WITH VISUAL SCORE BARS
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="gray")) as demo:
    gr.Markdown("# Resume ↔ Job Matcher")
    gr.Markdown("Upload or paste your **Resume** and **Job Description** to get an accurate match score with clear visual insights.")

    with gr.Row():
        with gr.Column():
            resume_input = gr.Textbox(label="Paste Resume Text", lines=10, placeholder="Paste your resume text here...")
            resume_pdf = gr.File(label="Upload Resume (PDF)", file_types=[".pdf"])
        with gr.Column():
            jd_input = gr.Textbox(label="Paste Job Description Text", lines=10, placeholder="Paste the job description here...")
            jd_pdf = gr.File(label="Upload Job Description (PDF)", file_types=[".pdf"])

    submit_btn = gr.Button("Compute Match Score", variant="primary")

    # core output – final score + recruiter-friendly feedback
    final_score_label = gr.Label(label="Final Match Score")
    feedback_box = gr.Textbox(label="Feedback & Suggestions", interactive=False, lines=4)

    # visual breakdown – semantic similarity and keyword scores as bars
    gr.Markdown("### Visual Breakdown of Scores")
    semantic_bar = gr.Slider(label="Semantic Similarity", minimum=0, maximum=1, step=0.01, interactive=False)
    keyword_bar = gr.Slider(label="Keyword Match", minimum=0, maximum=1, step=0.01, interactive=False)
    overall_bar = gr.Slider(label="Overall Match Score", minimum=0, maximum=1, step=0.01, interactive=False)

    # matched keywords
    matched_keywords_box = gr.Textbox(label="Matched Keywords", interactive=False)

    # spaced-out downloadable report
    gr.Markdown("<br>")
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
            download_report
        ]
    )

    gr.Markdown("This tool uses **semantic similarity + weighted keyword scoring** for high recruiter-level accuracy.")

demo.launch()