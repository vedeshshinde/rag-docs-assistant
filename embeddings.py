"""
embeddings.py
-------------
Turns text chunks into vectors ("embeddings") that can be compared for
similarity. This is the core idea behind RAG retrieval: instead of matching
keywords, we place text into a high-dimensional space where semantically
similar text ends up close together, then find nearest neighbors.

Two backends are supported:

1. Neural embeddings (preferred) - sentence-transformers' all-MiniLM-L6-v2,
   a small pretrained model that maps text to a 384-dim vector capturing
   meaning, not just word overlap. Needs internet on first run to download
   the model (~90MB), then works offline.

2. TF-IDF fallback - if sentence-transformers isn't installed or the model
   can't be downloaded (e.g. no internet / restricted network), we fall back
   to scikit-learn's TfidfVectorizer. This is a classical, keyword-frequency
   based vector representation - much weaker at capturing meaning/synonyms
   than neural embeddings, but fully local and dependency-light, so the
   pipeline still runs end to end. The vectorizer is fit once on the corpus
   at ingest time and pickled so queries reuse the exact same vector space.

Both paths expose the same interface: embed(texts) -> list of vectors.
This is intentional - it's the seam where you'd swap in OpenAI/Cohere/
Anthropic-compatible embedding APIs later without touching ingest.py or
query.py.
"""
import os
import pickle

VECTORIZER_PATH_DEFAULT = "./chroma_db/tfidf_vectorizer.pkl"


def _try_load_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model
    except Exception as e:
        print(f"[embeddings] Neural embedder unavailable ({e.__class__.__name__}: {e}).")
        print("[embeddings] Falling back to TF-IDF vectors.\n")
        return None


class Embedder:
    def __init__(self, vectorizer_path: str = VECTORIZER_PATH_DEFAULT):
        self.vectorizer_path = vectorizer_path
        self.model = _try_load_sentence_transformer()
        self.backend = "sentence-transformers" if self.model else "tfidf"
        self.vectorizer = None
        if self.backend == "tfidf" and os.path.exists(self.vectorizer_path):
            with open(self.vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)

    def fit_if_needed(self, corpus_texts):
        """For TF-IDF, must fit the vectorizer on the corpus before embedding.
        No-op for the neural backend, which needs no fitting."""
        if self.backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(max_features=2048)
            self.vectorizer.fit(corpus_texts)
            os.makedirs(os.path.dirname(self.vectorizer_path), exist_ok=True)
            with open(self.vectorizer_path, "wb") as f:
                pickle.dump(self.vectorizer, f)

    def embed(self, texts):
        if self.backend == "sentence-transformers":
            return self.model.encode(texts, show_progress_bar=False).tolist()
        else:
            if self.vectorizer is None:
                raise RuntimeError("TF-IDF vectorizer not fitted yet - call fit_if_needed() first.")
            return self.vectorizer.transform(texts).toarray().tolist()
