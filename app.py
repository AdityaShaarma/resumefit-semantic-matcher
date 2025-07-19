import gradio as gr
from model_utils import compute_final_score, generate_feedback, extract_keywords

def match_resume(resume_text, jd_text):
    if not resume_text.strip() or not jd_text.strip():
        return "Please provide both resume and job description.", "", "", ""

    final_score, semantic_score, keyword_score = compute_final_score(resume_text, jd_text)
    feedback = generate_feedback(resume_text, jd_text, semantic_score, keyword_score)
    matched_keywords = extract_keywords(resume_text) & extract_keywords(jd_text)

    return (
        f"{final_score * 100:.2f}%",
        f"Semantic Similarity: {semantic_score:.2f} | Keyword Match: {keyword_score:.2f}",
        feedback,
        ", ".join(sorted(matched_keywords)) if matched_keywords else "No matched technical keywords found."
    )

with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="gray")) as demo:
    gr.Markdown("# 📄 Resume ↔ Job Matcher")
    gr.Markdown("Upload or paste your **Resume** and **Job Description** below to see how well they align.")

    with gr.Row():
        resume_input = gr.Textbox(label="Paste Resume Text", lines=12, placeholder="Paste your resume text here...")
        jd_input = gr.Textbox(label="Paste Job Description", lines=12, placeholder="Paste the job description here...")

    with gr.Row():
        output_score = gr.Label(label="Final Match Score")
        output_details = gr.Textbox(label="Analysis Details", interactive=False)
    
    feedback_box = gr.Textbox(label="Feedback & Suggestions", interactive=False)
    matched_keywords_box = gr.Textbox(label="Matched Keywords", interactive=False)

    submit_btn = gr.Button("Compute Match Score", variant="primary")
    submit_btn.click(
        fn=match_resume,
        inputs=[resume_input, jd_input],
        outputs=[output_score, output_details, feedback_box, matched_keywords_box]
    )

    gr.Markdown("This tool uses semantic similarity + smart keyword extraction for accurate matching.")

demo.launch()