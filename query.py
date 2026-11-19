"""
query.py
--------
Retrieval-augmented question answering over the vector DB built by ingest.py.

Embeds the question with the SAME embedder used at ingest time, retrieves the
top-k most similar chunks, then asks Claude to answer using ONLY those
chunks, citing which source file(s) it used. If ANTHROPIC_API_KEY is not
set, falls back to printing the raw retrieved chunks so retrieval still
works standalone.

Usage:
    python query.py "What causes a hold time violation?"
    python query.py "How does routing relate to timing closure?" --k 4
"""
import argparse
import os

import chromadb

from embeddings import Embedder

SYSTEM_PROMPT = """You are a technical assistant answering questions using ONLY the
provided context chunks. Rules:
- Answer using only information present in the context below.
- If the context does not contain the answer, say so explicitly - do not guess.
- After your answer, list which source file(s) you used, e.g. "Sources: static_timing_analysis.txt".
- Be concise and technically precise.
"""


def retrieve(question: str, db_path: str, collection_name: str, k: int):
    embedder = Embedder(vectorizer_path=os.path.join(db_path, "tfidf_vectorizer.pkl"))
    query_vector = embedder.embed([question])[0]

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection_name, embedding_function=None)
    results = collection.query(query_embeddings=[query_vector], n_results=k)

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    return list(zip(docs, metadatas, distances))


def build_context(retrieved):
    blocks = []
    for doc, meta, _dist in retrieved:
        blocks.append(f"[Source: {meta['source']} | chunk {meta['chunk_index']}]\n{doc}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(question: str, context: str):
    try:
        import anthropic
    except ImportError:
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Context:\n\n{context}\n\nQuestion: {question}"
        }],
    )
    return message.content[0].text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--db_path", default="./chroma_db")
    parser.add_argument("--collection", default="docs")
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve")
    args = parser.parse_args()

    retrieved = retrieve(args.question, args.db_path, args.collection, args.k)

    print(f"\nRetrieved {len(retrieved)} chunks:")
    for doc, meta, dist in retrieved:
        preview = doc[:100].replace("\n", " ")
        print(f"  - {meta['source']} (chunk {meta['chunk_index']}, distance={dist:.3f}): {preview}...")

    context = build_context(retrieved)
    answer = generate_answer(args.question, context)

    print("\n" + "=" * 60)
    if answer:
        print("ANSWER (via Claude):\n")
        print(answer)
    else:
        print("No ANTHROPIC_API_KEY set (or anthropic package missing) -")
        print("showing retrieved context only. Set the key to get a generated answer:\n")
        print("  export ANTHROPIC_API_KEY=sk-ant-...\n")
        print(context)
    print("=" * 60)


if __name__ == "__main__":
    main()
