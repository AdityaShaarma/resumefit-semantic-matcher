import gradio as gr
import PyPDF2
import time
from model_utils import compute_final_score, generate_feedback, extract_keywords

# helper function to safely extract text from a PDF file
# ensures robust handling of pages with empty or malformed text
def extract_text_from_pdf(pdf_file):
    if not pdf_file:
        return ""
    extracted_text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            extracted_text += page.extract_text() or ""
            time.sleep(0.02)  # slight delay for smooth simulated progress
    except Exception:
        return ""
    return extracted_text.strip()

# main function that handles matching logic
# prioritizes PDF uploads over pasted text, validates input, and generates all outputs
def match_resume(resume_text, jd_text, resume_pdf, jd_pdf, progress=gr.Progress()):
    try:
        # prioritize PDF uploads if provided
        if resume_pdf:
            progress(0.1, desc="Extracting resume PDF...")
            resume_text = extract_text_from_pdf(resume_pdf)
        if jd_pdf:
            progress(0.2, desc="Extracting job description PDF...")
            jd_text = extract_text_from_pdf(jd_pdf)

        # validate inputs
        if not resume_text.strip() or not jd_text.strip():
            return (
                "Input Error",
                "Please provide both a resume and job description (paste text or upload PDFs).",
                0.0, 0.0, 0.0,
                "No keywords to display.",
                None
            )

        # compute semantic similarity and keyword match scores
        progress(0.5, desc="Analyzing semantic similarity and keyword relevance...")
        final_score, semantic_score, keyword_score = compute_final_score(resume_text, jd_text)

        # extract matched technical keywords
        progress(0.7, desc="Extracting matched technical keywords...")
        matched_keywords = extract_keywords(resume_text) & extract_keywords(jd_text)
        matched_keywords_text = (
            ", ".join(sorted(matched_keywords)) if matched_keywords else "No matched technical keywords found."
        )

        # generate actionable recruiter-style feedback
        progress(1.0, desc="Generating actionable feedback...")
        feedback = generate_feedback(resume_text, jd_text, semantic_score, keyword_score)

        # prepare downloadable match report
        report_content = (
            f"Resume ↔ Job Match Report\n\n"
            f"Final Match Score: {final_score * 100:.2f}%\n"
            f"Semantic Similarity: {semantic_score:.2f}\n"
            f"Keyword Match: {keyword_score:.2f}\n\n"
            f"Matched Keywords:\n{matched_keywords_text}\n\n"
            f"Feedback & Suggestions:\n{feedback}\n"
        )

        # return all processed outputs
        return (
            f"{final_score * 100:.2f}%",
            feedback,
            semantic_score,
            keyword_score,
            final_score,
            matched_keywords_text,
            report_content
        )

    except Exception as e:
        # catch-all error handling for unexpected issues
        return (
            "Processing Error",
            f"An unexpected error occurred while analyzing: {str(e)}",
            0.0, 0.0, 0.0,
            "No keywords to display due to an error.",
            None
        )

# define Gradio interface with a professional, recruiter-friendly design
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="gray")) as demo:
    # title and brief instructions
    gr.Markdown("# Resume ↔ Job Matcher")
    gr.Markdown(
        "Upload or paste your **Resume** and **Job Description** to get an accurate match score.\n"
        "The tool uses **semantic similarity + weighted keyword scoring** to mimic recruiter-level evaluation."
    )

    # input section: paste text or upload PDFs
    with gr.Row():
        with gr.Column():
            resume_input = gr.Textbox(
                label="Paste Resume Text", lines=10,
                placeholder="Paste your resume text here..."
            )
            resume_pdf = gr.File(
                label="Upload Resume (PDF)", file_types=[".pdf"]
            )
        with gr.Column():
            jd_input = gr.Textbox(
                label="Paste Job Description Text", lines=10,
                placeholder="Paste the job description here..."
            )
            jd_pdf = gr.File(
                label="Upload Job Description (PDF)", file_types=[".pdf"]
            )

    # action button
    submit_btn = gr.Button("Compute Match Score", variant="primary")

    # output section: core score and recruiter-style feedback
    final_score_label = gr.Label(label="Final Match Score")
    feedback_box = gr.Textbox(
        label="Feedback & Suggestions",
        interactive=False, lines=4
    )

    # visual breakdown: semantic similarity, keyword score, and overall score
    gr.Markdown("### Visual Breakdown of Scores")
    semantic_bar = gr.Slider(
        label="Semantic Similarity", minimum=0, maximum=1, step=0.01, interactive=False
    )
    keyword_bar = gr.Slider(
        label="Keyword Match", minimum=0, maximum=1, step=0.01, interactive=False
    )
    overall_bar = gr.Slider(
        label="Overall Match Score", minimum=0, maximum=1, step=0.01, interactive=False
    )

    # matched keywords display
    matched_keywords_box = gr.Textbox(
        label="Matched Keywords", interactive=False, lines=2
    )

    # spaced downloadable report button
    gr.Markdown("&nbsp;")  # adds spacing before download button
    download_report = gr.File(label="Download Feedback Report")

    # connect computation function to button
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

# launch the app
demo.launch()