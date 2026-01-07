"""
Gradio UI for Bible Semantic Similarity
Uses trained Word2Vec and Doc2Vec models for Romanian Bible text
"""
import gradio as gr
import spacy
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec
import os
import sys

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load spaCy model for preprocessing
print("Loading spaCy model...")
nlp = spacy.load("ro_core_news_sm", disable=['ner', 'parser'])

# Load trained models with error handling
print("Loading Word2Vec model...")
try:
    w2v_model = Word2Vec.load("bible_ro_word2vec.model")
    print(f"  Vocabulary size: {len(w2v_model.wv)}")
except Exception as e:
    print(f"  Error loading Word2Vec model: {e}")
    print("  The model may have been trained with a different Python/numpy version.")
    print("  Please retrain the model or use the same environment it was trained in.")
    w2v_model = None

print("Loading Doc2Vec model...")
try:
    d2v_model = Doc2Vec.load("bible_ro_doc2vec.model")
    print(f"  Document vectors: {len(d2v_model.dv)}")
except Exception as e:
    print(f"  Error loading Doc2Vec model: {e}")
    print("  The model may have been trained with a different Python/numpy version.")
    print("  Please retrain the model or use the same environment it was trained in.")
    d2v_model = None

if w2v_model:
    print("Word2Vec model loaded successfully!")
if d2v_model:
    print("Doc2Vec model loaded successfully!")


def preprocess(text: str) -> list[str]:
    """Preprocess text by tokenizing and lemmatizing."""
    doc = nlp(str(text))
    return [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]


def find_similar_words(word: str, top_n: int = 10) -> str:
    """Find similar words using Word2Vec."""
    if not word.strip():
        return "⚠️ Please enter a word."

    processed = preprocess(word)
    if not processed:
        return "⚠️ Could not process the word."

    query_word = processed[0]

    if query_word not in w2v_model.wv:
        return f"⚠️ Word '{query_word}' not found in vocabulary.\nTry a word that appears in the Bible corpus."

    similar_words = w2v_model.wv.most_similar(query_word, topn=top_n)

    result = f"🔍 **Query:** {word} → (lemma: {query_word})\n\n"
    result += "| Rank | Similar Word | Similarity |\n"
    result += "|------|--------------|------------|\n"

    for i, (similar_word, similarity) in enumerate(similar_words, 1):
        bar = "█" * int(similarity * 20)
        result += f"| {i} | {similar_word} | {similarity:.4f} {bar} |\n"

    return result


def compute_word_pair_similarity(word1: str, word2: str) -> str:
    """Compute similarity between two words."""
    if w2v_model is None:
        return "⚠️ Word2Vec model not available. The model may have been trained with a different Python/numpy version."

    if not word1.strip() or not word2.strip():
        return "⚠️ Please enter both words."

    processed1 = preprocess(word1)
    processed2 = preprocess(word2)

    if not processed1 or not processed2:
        return "⚠️ Could not preprocess one or both words."

    w1, w2 = processed1[0], processed2[0]

    if w1 not in w2v_model.wv:
        return f"⚠️ Word '{w1}' not found in vocabulary."
    if w2 not in w2v_model.wv:
        return f"⚠️ Word '{w2}' not found in vocabulary."

    similarity = w2v_model.wv.similarity(w1, w2)

    # Create visual representation
    bar_length = int(abs(similarity) * 30)
    bar = "█" * bar_length

    result = f"## Word Pair Similarity\n\n"
    result += f"**{word1}** ({w1}) ↔ **{word2}** ({w2})\n\n"
    result += f"### Similarity Score: **{similarity:.4f}**\n\n"
    result += f"```\n{bar} {similarity:.2%}\n```\n\n"

    # Interpretation
    if similarity > 0.8:
        result += "🟢 **Very High Similarity** - These words are nearly synonymous"
    elif similarity > 0.6:
        result += "🟢 **High Similarity** - These words are closely related"
    elif similarity > 0.4:
        result += "🟡 **Moderate Similarity** - These words share some semantic relationship"
    elif similarity > 0.2:
        result += "🟠 **Low Similarity** - These words are loosely related"
    else:
        result += "🔴 **Very Low Similarity** - These words have little semantic connection"

    return result


def find_similar_sentences(sentence: str, top_n: int = 10) -> str:
    """Find similar sentences using Doc2Vec."""
    if d2v_model is None:
        return "⚠️ Doc2Vec model not available. The model may have been trained with a different Python/numpy version."

    if not sentence.strip():
        return "⚠️ Please enter a sentence."

    tokens = preprocess(sentence)

    if not tokens:
        return "⚠️ Could not preprocess sentence."

    # Infer vector for input sentence
    inferred_vector = d2v_model.infer_vector(tokens)

    # Find most similar documents
    similar_docs = d2v_model.dv.most_similar([inferred_vector], topn=top_n)

    result = f"🔍 **Query:** {sentence}\n\n"
    result += f"**Preprocessed tokens:** {tokens}\n\n"
    result += "---\n\n"
    result += "### Most Similar Bible Passages\n\n"
    result += "| Rank | Document ID | Similarity |\n"
    result += "|------|-------------|------------|\n"

    for i, (tag, similarity) in enumerate(similar_docs, 1):
        bar = "█" * int(similarity * 20)
        result += f"| {i} | {tag} | {similarity:.4f} {bar} |\n"

    return result


def word_analogy(word_a: str, word_b: str, word_c: str, top_n: int = 5) -> str:
    """
    Perform word analogy: A is to B as C is to ?
    Example: "king" is to "man" as "queen" is to ?
    """
    if w2v_model is None:
        return "⚠️ Word2Vec model not available. The model may have been trained with a different Python/numpy version."

    if not all([word_a.strip(), word_b.strip(), word_c.strip()]):
        return "⚠️ Please enter all three words."

    proc_a = preprocess(word_a)
    proc_b = preprocess(word_b)
    proc_c = preprocess(word_c)

    if not proc_a or not proc_b or not proc_c:
        return "⚠️ Could not preprocess one or more words."

    a, b, c = proc_a[0], proc_b[0], proc_c[0]

    for word, lemma in [(word_a, a), (word_b, b), (word_c, c)]:
        if lemma not in w2v_model.wv:
            return f"⚠️ Word '{word}' ({lemma}) not found in vocabulary."

    try:
        # A - B + C = ?  =>  B is to A as C is to ?
        results = w2v_model.wv.most_similar(positive=[a, c], negative=[b], topn=top_n)

        result = f"## Word Analogy\n\n"
        result += f"**{word_a}** ({a}) is to **{word_b}** ({b}) as **{word_c}** ({c}) is to...\n\n"
        result += "| Rank | Word | Score |\n"
        result += "|------|------|-------|\n"

        for i, (word, score) in enumerate(results, 1):
            result += f"| {i} | **{word}** | {score:.4f} |\n"

        return result
    except Exception as e:
        return f"⚠️ Error computing analogy: {str(e)}"


# Create Gradio interface
with gr.Blocks(
    title="Bible Semantic Similarity",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
    )
) as demo:
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


if __name__ == "__main__":
    demo.launch()
