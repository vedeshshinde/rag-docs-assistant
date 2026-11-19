# Technical Docs RAG Assistant

A retrieval-augmented generation (RAG) pipeline that answers questions over a
set of technical documents, with citations back to the source file, and a
retrieval quality eval so answers aren't just "vibes-checked."

Built as a scoped, resume-ready project demonstrating: chunking, embeddings,
vector search, grounded LLM generation, and retrieval evaluation - the core
mechanics behind any production RAG or agentic system.

## Why this exists

Sample corpus (`sample_docs/`) covers three EDA/VLSI topics - static timing
analysis, place & route, and the RTL-to-GDSII flow - so the demo doubles as a
"documentation Q&A" tool, a common internal use case at chip-design and EDA
companies. Swap in your own PDFs/text files and it works the same way.

## Architecture

```
sample_docs/*.txt,*.pdf
        |
        v
   chunk_text()          <- ingest.py: splits into overlapping ~250-word chunks
        |
        v
   Embedder.embed()      <- embeddings.py: text -> vector
        |                    (sentence-transformers if available,
        |                     TF-IDF fallback otherwise - see below)
        v
   Chroma vector DB       <- persisted locally in ./chroma_db
        |
        v (at query time)
   embed(question) -> nearest-neighbor search -> top-k chunks
        |
        v
   Claude (Anthropic API) <- answers using ONLY retrieved chunks, cites sources
```

## Why two embedding backends?

`embeddings.py` tries `sentence-transformers` (a small neural model,
`all-MiniLM-L6-v2`) first - this is what "vector embeddings" usually means in
a RAG context: text mapped into a space where *meaning*, not just shared
keywords, determines closeness. If that model can't be downloaded (no
internet, restricted network, etc.), it falls back to scikit-learn TF-IDF
vectors - a much older, purely keyword-frequency-based vector representation.

This isn't just a robustness hack - it's worth understanding the difference
for an interview:
- **TF-IDF**: a chunk's vector is built purely from which words it contains
  and how rare those words are across the corpus. "Setup check" and "clock
  edge timing requirement" would look unrelated even though they mean similar
  things, because they share no words.
- **Neural embeddings**: the model has learned semantic relationships from
  training on huge amounts of text, so paraphrases and synonyms end up close
  together in vector space even with zero word overlap.

Run `python eval.py` with each backend and compare the precision numbers to
see this difference empirically, not just in theory.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # optional - see .env.example
```

## Usage

```bash
# 1. Build the vector index from sample_docs/ (or point --docs at your own folder)
python ingest.py --docs sample_docs --db_path ./chroma_db

# 2. Ask a question
python query.py "What is the difference between setup and hold checks?"

# 3. Measure retrieval quality against a small labeled test set
python eval.py
```

`eval.py` reports Precision@k: of the labeled questions, what fraction had
the correct source document show up in the top-k retrieved chunks. Edit
`TEST_SET` in `eval.py` to add your own labeled questions as you add
documents.

## Extending this

A few natural next steps, useful if you want to go deeper before an
interview:
- **Reranking**: retrieve top-20 with the vector search, then use a cheaper
  cross-encoder or the LLM itself to rerank down to the top-4 before
  generation - usually the single biggest quality lever in real RAG systems.
- **Agentic layer**: instead of always retrieving once, let Claude decide
  whether it needs to retrieve, retrieve again with a refined query, or
  answer directly - this is what "agentic RAG" means in practice.
- **Graph-based retrieval**: build a small knowledge graph (entities +
  relationships extracted from the docs) and retrieve by graph traversal
  for multi-hop questions ("what stage comes after the stage that checks
  DRC?") that pure similarity search struggles with.

## Files

| File            | Purpose                                              |
|-----------------|-------------------------------------------------------|
| `ingest.py`     | Chunk documents, embed them, store in Chroma          |
| `embeddings.py` | Shared embedding logic (neural + TF-IDF fallback)     |
| `query.py`      | Retrieve + generate a grounded, cited answer          |
| `eval.py`       | Precision@k retrieval evaluation on labeled questions |
| `sample_docs/`  | Example EDA/VLSI corpus (swap for your own docs)      |
