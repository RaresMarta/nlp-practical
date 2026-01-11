"""
Similarity services for Bible Semantic Similarity
Contains Word2Vec and Doc2Vec similarity functions
"""
from app.models import w2v_model, d2v_model
from app.preprocessing import preprocess
import numpy as np

from datasets import load_dataset

dataset = load_dataset("andyP/ro-paraphrase-bible")
text_columns = [col for col in dataset['train'].column_names if dataset['train'].features[col].dtype == 'string']
col1, col2 = text_columns[0], text_columns[1]
list1 = [s for s in dataset['train'][col1] if s is not None]
list2 = [s for s in dataset['train'][col2] if s is not None]
original_sentences_list = list(set(list1 + list2))


def get_model_info() -> dict:
    """Get detailed model information and statistics."""
    info = {
        "dataset_size": len(original_sentences_list),
        "total_training_pairs": len(dataset['train']),
        "w2v": {
            "vocab_size": len(w2v_model.wv) if w2v_model else 0,
            "vector_dim": w2v_model.vector_size if w2v_model else 0,
            "window": w2v_model.window if w2v_model else 0,
            "min_count": w2v_model.min_count if w2v_model else 0,
            "sg": w2v_model.sg if w2v_model else 0,
            "epochs": w2v_model.epochs if w2v_model else 0,
        },
        "d2v": {
            "doc_count": len(d2v_model.dv) if d2v_model else 0,
            "vector_dim": d2v_model.vector_size if d2v_model else 0,
            "window": d2v_model.window if d2v_model else 0,
            "min_count": d2v_model.min_count if d2v_model else 0,
            "epochs": d2v_model.epochs if d2v_model else 0,
        }
    }
    return info

def find_similar_words(word: str, top_n: int = 10) -> str:
    """Find similar words using Word2Vec with detailed statistics."""
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
    
    # Get word vector information
    word_vector = w2v_model.wv[query_word]
    vector_norm = np.linalg.norm(word_vector)
    vector_mean = np.mean(word_vector)
    vector_std = np.std(word_vector)

    result = f"🔍 **Query:** {word} → (lemma: {query_word})\n\n"
    
    # Add vector statistics
    result += "### Word Vector Statistics\n"
    result += f"- **Vector Dimension:** {len(word_vector)}\n"
    result += f"- **Vector Norm:** {vector_norm:.4f}\n"
    result += f"- **Mean Value:** {vector_mean:.6f}\n"
    result += f"- **Std Deviation:** {vector_std:.6f}\n\n"
    
    result += "### Top Similar Words\n"
    result += "| Rank | Similar Word | Similarity | Vector Norm |\n"
    result += "|------|--------------|------------|-------------|\n"

    for i, (similar_word, similarity) in enumerate(similar_words, 1):
        similar_vector = w2v_model.wv[similar_word]
        similar_norm = np.linalg.norm(similar_vector)
        bar = "█" * int(similarity * 20)
        result += f"| {i} | {similar_word} | {similarity:.4f} {bar} | {similar_norm:.4f} |\n"
    
    # Add model parameters
    result += "\n### Model Parameters\n"
    result += f"- **Model Type:** Word2Vec (CBOW) \n"
    result += f"- **Vector Dimension:** {w2v_model.vector_size}\n"
    result += f"- **Window Size:** {w2v_model.window}\n"
    result += f"- **Training Epochs:** {w2v_model.epochs}\n"
    result += f"- **Vocabulary Size:** {len(w2v_model.wv)}\n"

    return result


def compute_word_pair_similarity(word1: str, word2: str) -> str:
    """Compute similarity between two words with detailed analysis."""
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
    
    # Get detailed vector information
    vector1 = w2v_model.wv[w1]
    vector2 = w2v_model.wv[w2]
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    
    # Compute cosine distance and other metrics
    dot_product = np.dot(vector1, vector2)
    cosine_distance = 1 - similarity
    
    # Create visual representation
    bar_length = int(abs(similarity) * 30)
    bar = "█" * bar_length

    result = f"## Word Pair Similarity Analysis\n\n"
    result += f"**{word1}** ({w1}) ↔ **{word2}** ({w2})\n\n"
    
    result += "### Similarity Metrics\n"
    result += f"- **Cosine Similarity:** {similarity:.6f}\n"
    result += f"- **Cosine Distance:** {cosine_distance:.6f}\n"
    result += f"- **Dot Product:** {dot_product:.6f}\n"
    result += f"- **Vector 1 Norm:** {norm1:.6f}\n"
    result += f"- **Vector 2 Norm:** {norm2:.6f}\n\n"
    
    result += f"### Similarity Score: **{similarity:.4f}**\n\n"
    result += f"```\n{bar} {similarity:.2%}\n```\n\n"

    # Interpretation with thresholds
    if similarity > 0.8:
        result += "🟢 **Very High Similarity** - These words are nearly synonymous\n"
    elif similarity > 0.6:
        result += "🟢 **High Similarity** - These words are closely related\n"
    elif similarity > 0.4:
        result += "🟡 **Moderate Similarity** - These words share some semantic relationship\n"
    elif similarity > 0.2:
        result += "🟠 **Low Similarity** - These words are loosely related\n"
    else:
        result += "🔴 **Very Low Similarity** - These words have little semantic connection\n"
    
    # Add vector dimension info
    result += f"\n### Vector Information\n"
    result += f"- **Vector Dimension:** {len(vector1)}\n"
    result += f"- **Model:** Word2Vec (CBOW)\n"

    return result


def find_similar_sentences(sentence: str, top_n: int = 10) -> str:
    """Find similar sentences using Doc2Vec with detailed analysis."""
    if d2v_model is None:
        return "⚠️ Doc2Vec model not available."

    if not sentence.strip():
        return "⚠️ Please enter a sentence."

    tokens = preprocess(sentence)

    if not tokens:
        return "⚠️ Could not preprocess sentence."

    # Infer vector for input sentence
    inferred_vector = d2v_model.infer_vector(tokens)
    vector_norm = np.linalg.norm(inferred_vector)
    vector_mean = np.mean(inferred_vector)
    vector_std = np.std(inferred_vector)

    # Find most similar documents
    similar_docs = d2v_model.dv.most_similar([inferred_vector], topn=top_n)

    result = f"🔍 **Query:** {sentence}\n\n"
    
    # Add query vector statistics
    result += "### Query Vector Statistics\n"
    result += f"- **Tokens:** {len(tokens)}\n"
    result += f"- **Vector Dimension:** {len(inferred_vector)}\n"
    result += f"- **Vector Norm:** {vector_norm:.6f}\n"
    result += f"- **Mean Value:** {vector_mean:.6f}\n"
    result += f"- **Std Deviation:** {vector_std:.6f}\n\n"
    
    result += "---\n\n"
    result += "### Most Similar Bible Passages\n\n"

    result += "| Rank | Similarity | Doc Norm | Passage |\n"
    result += "|------|------------|----------|----------|\n"

    for i, (tag, similarity) in enumerate(similar_docs, 1):
        # Convert the tag back to an integer to index your list
        sentence_idx = int(tag)
        actual_sentence = original_sentences_list[sentence_idx]
        
        # Get document vector norm
        doc_vector = d2v_model.dv[int(tag)]
        doc_norm = np.linalg.norm(doc_vector)

        # Format the similarity bar
        bar = "█" * int(similarity * 10)
        
        # Truncate long passages for table display
        truncated_passage = actual_sentence[:80] + "..." if len(actual_sentence) > 80 else actual_sentence

        # Add row to table
        result += f"| {i} | {similarity:.4f} {bar} | {doc_norm:.4f} | {truncated_passage} |\n"
    
    # Add dataset and model info
    result += f"\n### Dataset & Model Statistics\n"
    result += f"- **Total Passages in Dataset:** {len(original_sentences_list)}\n"
    result += f"- **Model Type:** Doc2Vec (Paragraph Vectors)\n"
    result += f"- **Vector Dimension:** {d2v_model.vector_size}\n"
    result += f"- **Window Size:** {d2v_model.window}\n"
    result += f"- **Training Epochs:** {d2v_model.epochs}\n"

    return result


def word_analogy(word_a: str, word_b: str, word_c: str, top_n: int = 5) -> str:
    """
    Perform word analogy with detailed analysis: A is to B as C is to ?
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
        
        # Compute vector relationships
        vec_a = w2v_model.wv[a]
        vec_b = w2v_model.wv[b]
        vec_c = w2v_model.wv[c]
        
        # Vector difference (a - b + c)
        analogy_vector = vec_a - vec_b + vec_c
        analogy_norm = np.linalg.norm(analogy_vector)

        result = f"## Word Analogy Analysis\n\n"
        result += f"**{word_a}** ({a}) is to **{word_b}** ({b}) as **{word_c}** ({c}) is to...\n\n"
        
        result += "### Analogy Vector Statistics\n"
        result += f"- **Vector A - B + C Norm:** {analogy_norm:.6f}\n"
        result += f"- **Vector Mean:** {np.mean(analogy_vector):.6f}\n"
        result += f"- **Vector Std:** {np.std(analogy_vector):.6f}\n\n"
        
        result += "### Top Results\n"
        result += "| Rank | Word | Analogy Score | Vector Norm |\n"
        result += "|------|------|----------------|-------------|\n"

        for i, (word, score) in enumerate(results, 1):
            result_vector = w2v_model.wv[word]
            result_norm = np.linalg.norm(result_vector)
            result += f"| {i} | **{word}** | {score:.4f} | {result_norm:.4f} |\n"
        
        result += f"\n### Model Information\n"
        result += f"- **Vector Dimension:** {w2v_model.vector_size}\n"
        result += f"- **Vocabulary Size:** {len(w2v_model.wv)}\n"

        return result
    except Exception as e:
        return f"⚠️ Error computing analogy: {str(e)}"
