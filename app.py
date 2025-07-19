import gradio as gr
import PyPDF2
from model_utils import (
    get_embedding, compute_similarity, keyword_overlap,
    compute_final_score, generate_feedback, extract_skills_ner
)

# PDF TEXT EXTRACTION
def extract_text_from_pdf(file_obj) -> str:
    # Extracts text from uploaded PDF; returns empty string if extraction fails
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(file_obj)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception:
        return ""

# MAIN COMPUTATION FUNCTION
def compute_match(resume_input, jd_input, resume_pdf=None, jd_pdf=None):
    # Extract text from PDFs if uploaded, otherwise use text inputs
    resume_text = extract_text_from_pdf(resume_pdf) if resume_pdf else resume_input
    jd_text = extract_text_from_pdf(jd_pdf) if jd_pdf else jd_input

    if not resume_text or not jd_text:
        return "Please provide both Resume and Job Description.", "", [], []

    # Compute similarity scores
    resume_vec = get_embedding(resume_text)
    jd_vec = get_embedding(jd_text)
    embedding_score = compute_similarity(resume_vec, jd_vec)
    keyword_score = keyword_overlap(resume_text, jd_text)
    final_score = compute_final_score(embedding_score, keyword_score)

    # Generate feedback and matched keywords
    feedback = generate_feedback(resume_text, jd_text, embedding_score, keyword_score)
    matched_keywords = sorted(list(extract_skills_ner(resume_text) & extract_skills_ner(jd_text)))

    return (
        f"{final_score * 100:.2f}%",  # Final score
        "\n".join(feedback),         # Feedback text
        matched_keywords,            # Keywords to show as tags
        feedback                     # For downloadable report
    )

# GRADIO INTERFACE DESIGN
with gr.Blocks(title="ResumeFit Semantic Matcher") as demo:
    gr.Markdown("# ✅ Resume ↔ Job Matcher\nUpload your resume and job description to check compatibility.")

    with gr.Row():
        with gr.Column():
            resume_input = gr.Textbox(label="Paste Resume Text", lines=10, placeholder="Paste your resume here...")
            resume_pdf = gr.File(label="Upload Resume (PDF)", file_types=[".pdf"])
        with gr.Column():
            jd_input = gr.Textbox(label="Paste Job Description Text", lines=10, placeholder="Paste job description here...")
            jd_pdf = gr.File(label="Upload Job Description (PDF)", file_types=[".pdf"])

    compute_btn = gr.Button("Compute Match Score", variant="primary")

    final_score = gr.Label(label="Final Match Score")
    feedback_box = gr.Textbox(label="Why You Got This Score", lines=4, interactive=False)
    matched_keywords = gr.HighlightedText(label="Matched Keywords")
    download_feedback = gr.File(label="Download Feedback Report", type="file")

    def save_feedback_for_download(feedback_list):
        with open("feedback_report.txt", "w") as f:
            f.write("\n".join(feedback_list))
        return "feedback_report.txt"

    compute_btn.click(
        fn=compute_match,
        inputs=[resume_input, jd_input, resume_pdf, jd_pdf],
        outputs=[final_score, feedback_box, matched_keywords, download_feedback],
        postprocess=lambda final, fb, kw, fb_list: (final, fb, [(kw, "match") for kw in kw], save_feedback_for_download(fb_list))
    )

# RUN APP LOCALLY OR ON HUGGING FACE
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)