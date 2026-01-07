"""
Script to load trained Word2Vec and Doc2Vec models and test similarity.
"""
import spacy
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec


# Load spaCy model for preprocessing
nlp = spacy.load("ro_core_news_sm", disable=['ner', 'parser'])


def preprocess(text: str) -> list[str]:
    """
    Preprocess text by tokenizing and lemmatizing.
    Returns a list of lowercase lemmas, excluding punctuation and whitespace.
    """
    doc = nlp(str(text))
    return [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]


def load_models(w2v_path: str = "bible_ro_word2vec.model",
                d2v_path: str = "bible_ro_doc2vec.model"):
    """
    Load the trained Word2Vec and Doc2Vec models.

    Args:
        w2v_path: Path to the Word2Vec model file
        d2v_path: Path to the Doc2Vec model file

    Returns:
        Tuple of (Word2Vec model, Doc2Vec model)
    """
    print("Loading Word2Vec model...")
    w2v_model = Word2Vec.load(w2v_path)
    print(f"  Vocabulary size: {len(w2v_model.wv)}")

    print("Loading Doc2Vec model...")
    d2v_model = Doc2Vec.load(d2v_path)
    print(f"  Document vectors: {len(d2v_model.dv)}")

    return w2v_model, d2v_model


def test_word_similarity(w2v_model, word: str, topn: int = 10):
    """
    Find the most similar words to a given word using Word2Vec.

    Args:
        w2v_model: Trained Word2Vec model
        word: The word to find similar words for
        topn: Number of similar words to return
    """
    print(f"\n{'='*60}")
    print(f"Word Similarity for: '{word}'")
    print('='*60)

    # Preprocess the word (lemmatize it)
    processed = preprocess(word)
    if not processed:
        print("Could not process the word.")
        return

    query_word = processed[0]
    print(f"Preprocessed query: '{query_word}'")

    if query_word not in w2v_model.wv:
        print(f"Word '{query_word}' not found in vocabulary.")
        print("Try a word that appears at least twice in the training corpus.")
        return

    similar_words = w2v_model.wv.most_similar(query_word, topn=topn)

    print(f"\nTop {topn} similar words:")
    print("-" * 40)
    for i, (similar_word, similarity) in enumerate(similar_words, 1):
        print(f"{i:2}. {similar_word:20} (similarity: {similarity:.4f})")


def test_sentence_similarity(d2v_model, sentence: str, original_sentences: list[str] = None, topn: int = 10):
    """
    Find the most similar sentences/documents using Doc2Vec.

    Args:
        d2v_model: Trained Doc2Vec model
        sentence: The sentence to find similar sentences for
        original_sentences: Optional list of original sentences (to display the actual text)
        topn: Number of similar sentences to return
    """
    print(f"\n{'='*60}")
    print(f"Sentence Similarity for: '{sentence}'")
    print('='*60)

    # Preprocess the sentence
    tokens = preprocess(sentence)
    print(f"Preprocessed tokens: {tokens}")

    if not tokens:
        print("Could not preprocess sentence.")
        return

    # Infer the vector for the input sentence
    inferred_vector = d2v_model.infer_vector(tokens)

    # Find most similar documents
    similar_docs = d2v_model.dv.most_similar([inferred_vector], topn=topn)

    print(f"\nTop {topn} similar sentences:")
    print("-" * 40)
    for i, (tag, similarity) in enumerate(similar_docs, 1):
        if original_sentences is not None:
            idx = int(tag)
            if idx < len(original_sentences):
                text = original_sentences[idx]
                # Truncate long sentences for display
                display_text = text[:100] + "..." if len(text) > 100 else text
                print(f"{i:2}. [{similarity:.4f}] {display_text}")
            else:
                print(f"{i:2}. [{similarity:.4f}] Document tag: {tag}")
        else:
            print(f"{i:2}. [{similarity:.4f}] Document tag: {tag}")


def compute_word_pair_similarity(w2v_model, word1: str, word2: str):
    """
    Compute similarity between two words.
    """
    print(f"\n{'='*60}")
    print(f"Similarity between '{word1}' and '{word2}'")
    print('='*60)

    processed1 = preprocess(word1)
    processed2 = preprocess(word2)

    if not processed1 or not processed2:
        print("Could not preprocess one or both words.")
        return

    w1, w2 = processed1[0], processed2[0]

    if w1 not in w2v_model.wv:
        print(f"Word '{w1}' not found in vocabulary.")
        return
    if w2 not in w2v_model.wv:
        print(f"Word '{w2}' not found in vocabulary.")
        return

    similarity = w2v_model.wv.similarity(w1, w2)
    print(f"Similarity: {similarity:.4f}")


def interactive_mode(w2v_model, d2v_model):
    """
    Interactive mode for testing similarity.
    """
    print("\n" + "="*60)
    print("INTERACTIVE SIMILARITY TESTING")
    print("="*60)
    print("\nCommands:")
    print("  w <word>         - Find similar words")
    print("  wp <word1> <word2> - Compute similarity between two words")
    print("  s <sentence>     - Find similar sentences")
    print("  q                - Quit")
    print()

    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'q':
                print("Goodbye!")
                break

            if user_input.startswith('w '):
                word = user_input[2:].strip()
                if ' ' in word:
                    # Word pair similarity
                    parts = word.split(maxsplit=1)
                    if len(parts) == 2:
                        compute_word_pair_similarity(w2v_model, parts[0], parts[1])
                else:
                    test_word_similarity(w2v_model, word)

            elif user_input.startswith('wp '):
                parts = user_input[3:].strip().split()
                if len(parts) >= 2:
                    compute_word_pair_similarity(w2v_model, parts[0], parts[1])
                else:
                    print("Usage: wp <word1> <word2>")

            elif user_input.startswith('s '):
                sentence = user_input[2:].strip()
                test_sentence_similarity(d2v_model, sentence)

            else:
                print("Unknown command. Use 'w', 'wp', 's', or 'q'.")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """
    Main function to run similarity tests.
    """
    import os

    # Change to script directory if models are there
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Load models
    try:
        w2v_model, d2v_model = load_models()
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        print("Make sure the model files are in the current directory.")
        return

    print("\n" + "="*60)
    print("MODELS LOADED SUCCESSFULLY")
    print("="*60)

    # Example tests
    print("\n" + "#"*60)
    print("# WORD SIMILARITY EXAMPLES")
    print("#"*60)

    # Test words from the Bible corpus
    test_words = ["Dumnezeu", "iubire", "credință", "păcat"]
    for word in test_words:
        test_word_similarity(w2v_model, word, topn=5)

    # Word pair similarity
    print("\n" + "#"*60)
    print("# WORD PAIR SIMILARITY")
    print("#"*60)
    compute_word_pair_similarity(w2v_model, "Dumnezeu", "Domnul")
    compute_word_pair_similarity(w2v_model, "iubire", "dragoste")
    compute_word_pair_similarity(w2v_model, "bine", "rău")

    # Sentence similarity
    print("\n" + "#"*60)
    print("# SENTENCE SIMILARITY EXAMPLES")
    print("#"*60)

    test_sentences = [
        "La început era Cuvântul",
        "Dumnezeu este iubire",
        "Cuvântul este adevăr"
    ]
    for sentence in test_sentences:
        test_sentence_similarity(d2v_model, sentence, topn=5)

    # Start interactive mode
    print("\n" + "#"*60)
    print("# ENTERING INTERACTIVE MODE")
    print("#"*60)
    interactive_mode(w2v_model, d2v_model)


if __name__ == "__main__":
    main()
