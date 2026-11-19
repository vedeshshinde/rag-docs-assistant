"""
ingest.py
---------
Reads all .txt and .pdf files from a docs folder, splits them into overlapping
chunks, embeds each chunk (see embeddings.py), and stores the chunks +
vectors in a local persistent Chroma vector database.

Usage:
    python ingest.py --docs sample_docs --db_path ./chroma_db
"""
import argparse
import glob
import os

import chromadb
from pypdf import PdfReader

from embeddings import Embedder


def load_text(path: str) -> str:
    """Load raw text from a .txt or .pdf file."""
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = 250, overlap: int = 50):
    """
    Split text into overlapping word-based chunks.

    Smaller chunks (default 250 words) generally retrieve more precisely for
    Q&A - large chunks bury the answer in irrelevant surrounding text.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap  # step forward, but re-include the overlap
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="sample_docs", help="Folder of .txt/.pdf files")
    parser.add_argument("--db_path", default="./chroma_db", help="Where to persist the vector DB")
    parser.add_argument("--chunk_size", type=int, default=250)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--collection", default="docs")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.docs, "*.txt")) +
                    glob.glob(os.path.join(args.docs, "*.pdf")))
    if not files:
        print(f"No .txt or .pdf files found in {args.docs}/")
        return

    all_ids, all_docs, all_metadatas = [], [], []
    for path in files:
        filename = os.path.basename(path)
        text = load_text(path)
        chunks = chunk_text(text, args.chunk_size, args.overlap)
        print(f"{filename}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            all_ids.append(f"{filename}::{i}")
            all_docs.append(chunk)
            all_metadatas.append({"source": filename, "chunk_index": i})

    print("\nComputing embeddings...")
    embedder = Embedder(vectorizer_path=os.path.join(args.db_path, "tfidf_vectorizer.pkl"))
    embedder.fit_if_needed(all_docs)  # no-op for neural backend
    vectors = embedder.embed(all_docs)
    print(f"Backend used: {embedder.backend} ({len(vectors[0])}-dim vectors)")

    client = chromadb.PersistentClient(path=args.db_path)
    try:
        client.delete_collection(args.collection)
    except Exception:
        pass
    # embedding_function=None: we pass precomputed vectors ourselves below,
    # so Chroma never tries to download/run its own default embedder.
    collection = client.create_collection(args.collection, embedding_function=None)
    collection.add(ids=all_ids, documents=all_docs, metadatas=all_metadatas, embeddings=vectors)

    print(f"\nIngested {len(all_docs)} chunks from {len(files)} files into '{args.db_path}'.")
    print(f"Embedding backend: {embedder.backend}")


if __name__ == "__main__":
    main()
