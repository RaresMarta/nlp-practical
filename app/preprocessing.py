"""
Text preprocessing for Bible Semantic Similarity
"""
from app.models import nlp


def preprocess(text: str) -> list[str]:
    """Preprocess text by tokenizing and lemmatizing."""
    doc = nlp(str(text))
    return [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
