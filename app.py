"""
Gradio UI for Bible Semantic Similarity
Uses trained Word2Vec and Doc2Vec models for Romanian Bible text
"""
from app.ui import create_demo

if __name__ == "__main__":
    demo = create_demo()
    # Apply theme at launch (Gradio 6.0+ requirement)
    theme = getattr(demo, '_custom_theme', None)
    if theme:
        demo.launch(theme=theme)
    else:
        demo.launch()
