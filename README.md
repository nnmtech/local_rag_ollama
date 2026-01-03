Ollama + Chroma small RAG demo
==============================

This repository contains a small example script at `venv/app.py` that:
- uses Ollama for embeddings and chat (local LLM backend), and
- uses a persistent ChromaDB collection for retrieval.

Quick setup
-----------
1. Copy `.env.example` to `venv/.env` (edit values):

   cp .env.example venv/.env

2. Ensure Ollama is running locally (default `http://localhost:11434`) and that the models configured in `venv/.env` are available.

Run
---
From the repository root run:

```bash
./venv/bin/python venv/app.py
```

Note: `venv/app.py` executes indexing + queries at import time (no `if __name__ == "__main__"` guard). Avoid importing it from other Python modules unless you intend it to run.

What the script does
---------------------
- Loads text files from `./news_articles` (create that directory and add `.txt` files).
- Splits documents into chunks.
- Generates semantic embeddings via Ollama (`POST $OLLAMA_HOST/api/embeddings`).
- Upserts chunks into a persistent Chroma DB in `./chroma_persistent_storage`.
- Queries the vector DB and asks Ollama chat for a final answer using retrieved context.

Environment variables
---------------------
- `OLLAMA_HOST` (optional) — Ollama host (default `http://localhost:11434`).
- `OLLAMA_CHAT_MODEL` (optional) — chat model (default `llama3.2:3b`).
- `OLLAMA_EMBED_MODEL` (optional) — embedding model (default `nomic-embed-text:latest`).
- `CHROMA_COLLECTION` (optional) — overrides the collection name; defaults to `rag_{OLLAMA_EMBED_MODEL}` (sanitized) to avoid dimension mismatch when switching embedding models.
- `DEBUG=1` or `PERPLEXITY_DEBUG=1` (optional) — prints HTTP payloads/responses (can be large; may include full prompts/retrieved context).

Notes & best practices
----------------------
- Do not commit secrets. Keep `venv/.env` out of version control.
- Prefer switching collections (set `CHROMA_COLLECTION`, or change `OLLAMA_EMBED_MODEL`) instead of deleting Chroma data.
- For a full clean reset (recreated on next run):

   ```bash
   test -d ./chroma_persistent_storage && rm -rf ./chroma_persistent_storage
   ```

If you want, I can:
- add a `--no-index` mode to speed up repeated runs, or
- wrap execution in a `main()` + `if __name__ == "__main__"` guard.
