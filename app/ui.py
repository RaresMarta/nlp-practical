"""
Gradio UI for Bible Semantic Similarity
"""
import gradio as gr
from app.services import (
    find_similar_words,
    compute_word_pair_similarity,
    find_similar_sentences,
    word_analogy,
)


def create_demo() -> gr.Blocks:
    """Create and return the Gradio Blocks demo interface."""
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
    )

    with gr.Blocks(title="Bible Semantic Similarity") as demo:
        gr.Markdown("""
        # 📖 Bible Semantic Similarity Explorer

        Explore semantic relationships in Romanian Bible text using Word2Vec and Doc2Vec models.

        **Models trained on:** Romanian Bible Paraphrase Dataset (~247,000 sentences)
        """)

        with gr.Tabs():
            # Tab 1: Similar Words
            with gr.TabItem("🔤 Similar Words"):
                gr.Markdown("### Find words with similar meanings")
                with gr.Row():
                    with gr.Column(scale=1):
                        word_input = gr.Textbox(
                            label="Enter a Romanian word",
                            placeholder="e.g., Dumnezeu, iubire, credință, păcat",
                            lines=1
                        )
                        word_top_n = gr.Slider(
                            minimum=1, maximum=20, value=10, step=1,
                            label="Number of results"
                        )
                        word_btn = gr.Button("Find Similar Words", variant="primary")
                    with gr.Column(scale=2):
                        word_output = gr.Markdown(label="Results")

                word_btn.click(
                    find_similar_words,
                    inputs=[word_input, word_top_n],
                    outputs=word_output
                )

                gr.Examples(
                    examples=[
                        ["Dumnezeu", 10],
                        ["iubire", 10],
                        ["credință", 10],
                        ["păcat", 10],
                        ["binecuvântare", 10],
                    ],
                    inputs=[word_input, word_top_n],
                )

            # Tab 2: Word Pair Similarity
            with gr.TabItem("⚖️ Word Pair Similarity"):
                gr.Markdown("### Compare similarity between two words")
                with gr.Row():
                    with gr.Column(scale=1):
                        pair_word1 = gr.Textbox(
                            label="First word",
                            placeholder="e.g., Dumnezeu"
                        )
                        pair_word2 = gr.Textbox(
                            label="Second word",
                            placeholder="e.g., Domnul"
                        )
                        pair_btn = gr.Button("Compare Words", variant="primary")
                    with gr.Column(scale=2):
                        pair_output = gr.Markdown(label="Results")

                pair_btn.click(
                    compute_word_pair_similarity,
                    inputs=[pair_word1, pair_word2],
                    outputs=pair_output
                )

                gr.Examples(
                    examples=[
                        ["Dumnezeu", "Domnul"],
                        ["iubire", "dragoste"],
                        ["bine", "rău"],
                        ["cer", "pământ"],
                        ["înger", "diavol"],
                    ],
                    inputs=[pair_word1, pair_word2],
                )

            # Tab 3: Word Analogy
            with gr.TabItem("🧮 Word Analogy"):
                gr.Markdown("### Word analogies: A is to B as C is to ?")
                with gr.Row():
                    with gr.Column(scale=1):
                        analogy_a = gr.Textbox(label="Word A", placeholder="e.g., rege")
                        analogy_b = gr.Textbox(label="Word B", placeholder="e.g., bărbat")
                        analogy_c = gr.Textbox(label="Word C", placeholder="e.g., regină")
                        analogy_top_n = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="Number of results"
                        )
                        analogy_btn = gr.Button("Find Analogy", variant="primary")
                    with gr.Column(scale=2):
                        analogy_output = gr.Markdown(label="Results")

                analogy_btn.click(
                    word_analogy,
                    inputs=[analogy_a, analogy_b, analogy_c, analogy_top_n],
                    outputs=analogy_output
                )

                gr.Examples(
                    examples=[
                        ["tatăl", "fiul", "mama", 5],
                        ["ceruri", "Dumnezeu", "pământ", 5],
                    ],
                    inputs=[analogy_a, analogy_b, analogy_c, analogy_top_n],
                )

            # Tab 4: Sentence Similarity
            with gr.TabItem("📝 Sentence Similarity"):
                gr.Markdown("### Find similar Bible passages")
                with gr.Row():
                    with gr.Column(scale=1):
                        sentence_input = gr.Textbox(
                            label="Enter a sentence in Romanian",
                            placeholder="e.g., La început era Cuvântul",
                            lines=3
                        )
                        sentence_top_n = gr.Slider(
                            minimum=1, maximum=20, value=10, step=1,
                            label="Number of results"
                        )
                        sentence_btn = gr.Button("Find Similar Passages", variant="primary")
                    with gr.Column(scale=2):
                        sentence_output = gr.Markdown(label="Results")

                sentence_btn.click(
                    find_similar_sentences,
                    inputs=[sentence_input, sentence_top_n],
                    outputs=sentence_output
                )

                gr.Examples(
                    examples=[
                        ["La început era Cuvântul", 10],
                        ["Dumnezeu este iubire", 10],
                        ["Cuvântul este adevăr", 10],
                        ["Iisus a spus", 10],
                    ],
                    inputs=[sentence_input, sentence_top_n],
                )

        gr.Markdown("""
        ---
        **Note:** This tool uses Word2Vec for word-level similarity and Doc2Vec for sentence-level similarity.
        Words are lemmatized before lookup (e.g., "Dumnezeu" → "dumnezeu").
        """)

    # Store theme for launch
    demo._custom_theme = theme
    return demo
