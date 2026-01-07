"""
Similarity services for Bible Semantic Similarity
Contains Word2Vec and Doc2Vec similarity functions
"""
from app.models import w2v_model, d2v_model
from app.preprocessing import preprocess


def find_similar_words(word: str, top_n: int = 10) -> str:
    """Find similar words using Word2Vec."""
    if w2v_model is None:
        return "⚠️ Word2Vec model not available. The model may have been trained with a different Python/numpy version."

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
