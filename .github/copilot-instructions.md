# Copilot instructions (RAG demo)

## Big picture (source of truth)
- Single-file RAG pipeline in [venv/app.py](../venv/app.py):
  1) load `.txt` from [news_articles/](../news_articles/)
  2) chunk text (`split_text(chunk_size=1000, chunk_overlap=20)`)
  3) embed chunks via Ollama (`POST $OLLAMA_HOST/api/embeddings`)
  4) upsert into persistent Chroma at [chroma_persistent_storage/](../chroma_persistent_storage/)
  5) retrieve chunks (`collection.query`)
  6) answer via Ollama chat (`POST $OLLAMA_HOST/api/chat`)

## How to run (local)
- Preferred (if the local venv exists): `./venv/bin/python venv/app.py`
- Prereqs:
  - Ollama running (default `http://localhost:11434`)
  - Models available: `$OLLAMA_CHAT_MODEL` (default `llama3.2:3b`), `$OLLAMA_EMBED_MODEL` (default `nomic-embed-text:latest`)
- Note: the script runs indexing + queries at import time (no `if __name__ == "__main__"` guard). Avoid importing it from other modules unless you intend it to execute.

## Environment & configuration
- Env vars are loaded from [venv/.env](../venv/.env) (relative to the script, not repo root).
- Variables used by the code:
  - `OLLAMA_HOST`, `OLLAMA_CHAT_MODEL`, `OLLAMA_EMBED_MODEL`
  - `CHROMA_COLLECTION` (optional; default derived from embedding model)
  - `DEBUG=1` or `PERPLEXITY_DEBUG=1` prints HTTP payloads/responses (can be large; may include full prompts/retrieved context)
- Treat secrets as sensitive: don’t print or commit keys from `venv/.env`.

## Chroma conventions (important)
- Persistent client: `chromadb.PersistentClient(path="./chroma_persistent_storage")`.
- Collection naming is embedding-model-aware: defaults to `rag_{OLLAMA_EMBED_MODEL}` (sanitized). This prevents dimension mismatch when switching embedding models.
- If you change embedding models/backends, use a new collection (set `CHROMA_COLLECTION`) or clear [chroma_persistent_storage/](../chroma_persistent_storage/) locally.

## Reset local vector DB
- Prefer switching collections (set `CHROMA_COLLECTION`, or change `OLLAMA_EMBED_MODEL`) to avoid deleting data.
- For a full clean reset (recreated on next run):
  - `test -d ./chroma_persistent_storage && rm -rf ./chroma_persistent_storage`

## Data/layout conventions
- Corpus: `.txt` files in [news_articles/](../news_articles/); filename becomes the document id.
- Chunk ids: `${filename}_chunkN`; `collection.upsert` makes reruns id-stable (same ids overwrite).

## Project-specific editing guidance
- Keep request timeouts and error formatting consistent with `_http_error_message` in [venv/app.py](../venv/app.py).
- This repo’s README is currently out of date (mentions Perplexity/local embeddings). Prefer the code and this file as the authoritative behavior.
- Avoid scanning/modifying the whole virtualenv; treat [venv/app.py](../venv/app.py) as the owned source file.
