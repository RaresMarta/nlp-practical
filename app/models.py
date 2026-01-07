"""
Model loading for Bible Semantic Similarity
Loads spaCy, Word2Vec, and Doc2Vec models
"""
import os
import spacy
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec

# Get the project root directory (parent of app/)
_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load spaCy model for preprocessing
print("Loading spaCy model...")
nlp = spacy.load("ro_core_news_sm", disable=['ner', 'parser'])

# Load trained models with error handling
print("Loading Word2Vec model...")
try:
    w2v_model = Word2Vec.load(os.path.join(_project_dir, "bible_ro_word2vec.model"))
    print(f"  Vocabulary size: {len(w2v_model.wv)}")
except Exception as e:
    print(f"  Error loading Word2Vec model: {e}")
    print("  The model may have been trained with a different Python/numpy version.")
    print("  Please retrain the model or use the same environment it was trained in.")
    w2v_model = None

print("Loading Doc2Vec model...")
try:
    d2v_model = Doc2Vec.load(os.path.join(_project_dir, "bible_ro_doc2vec.model"))
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
