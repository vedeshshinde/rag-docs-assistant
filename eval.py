"""
eval.py
-------
Measures retrieval quality: for a small labeled set of (question, correct_source)
pairs, checks whether the correct source file appears in the top-k retrieved
chunks. Reports precision@k for several values of k.

This is the piece that turns "I built a RAG demo" into "I measured and tuned
retrieval quality" - worth walking through in an interview.

Usage:
    python eval.py
"""
import os

import chromadb

from embeddings import Embedder

# Each entry: a question, and the source file that actually contains the answer.
# Edit / extend this list as you add more documents.
TEST_SET = [
    ("What is the difference between setup and hold checks?", "static_timing_analysis.txt"),
    ("Why is slack important in timing analysis?", "static_timing_analysis.txt"),
    ("How is the critical path found in a timing graph?", "static_timing_analysis.txt"),
    ("What does analytical placement try to minimize?", "place_and_route.txt"),
    ("What algorithm is detailed routing similar to?", "place_and_route.txt"),
    ("What is the difference between global and detailed routing?", "place_and_route.txt"),
    ("What is checked during physical verification?", "rtl_to_gdsii_flow.txt"),
    ("What does an EDA team's internal tooling typically automate?", "rtl_to_gdsii_flow.txt"),
    ("What is GDSII?", "rtl_to_gdsii_flow.txt"),
]


def evaluate(db_path="./chroma_db", collection_name="docs", k_values=(1, 3, 5)):
    embedder = Embedder(vectorizer_path=os.path.join(db_path, "tfidf_vectorizer.pkl"))
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection_name, embedding_function=None)

    max_k = max(k_values)
    hits_at_k = {k: 0 for k in k_values}

    for question, correct_source in TEST_SET:
        query_vector = embedder.embed([question])[0]
        results = collection.query(query_embeddings=[query_vector], n_results=max_k)
        retrieved_sources = [m["source"] for m in results["metadatas"][0]]
        for k in k_values:
            if correct_source in retrieved_sources[:k]:
                hits_at_k[k] += 1

    print(f"Embedding backend: {embedder.backend}")
    print(f"Evaluated on {len(TEST_SET)} labeled questions.\n")
    for k in k_values:
        precision = hits_at_k[k] / len(TEST_SET)
        print(f"  Precision@{k}: {precision:.2f}  ({hits_at_k[k]}/{len(TEST_SET)})")


if __name__ == "__main__":
    evaluate()
