PerplexiPy + Chroma small RAG demo
=================================

This repository contains a small example script at `venv/app.py` that:
- uses Perplexity (API) as the LLM backend, and
- uses a deterministic local embedding function by default to avoid requiring OpenAI keys.

Quick setup
-----------
1. Copy `.env.example` to `.env` (edit values):

   cp .env.example venv/.env

2. Edit `venv/.env` and set `PERPLEXITY_API_KEY` to your Perplexity API key.
   Optionally set `PERPLEXITY_MODEL` (defaults to `sonar`).

Run
---
From the repository root run:

```bash
python3 venv/app.py
```

What the script does
---------------------
- Loads text files from `./news_articles` (create that directory and add `.txt` files).
- Splits documents into chunks and creates a Chroma collection `perplexipy_vc_collection`.
- Generates deterministic local embeddings (SHA256-based) for offline testing.
- Queries the vector DB and asks Perplexity for a final answer using the provided context.

Environment variables
---------------------
- `PERPLEXITY_API_KEY` (required) — your Perplexity API key (starts with `pplx-`).
- `PERPLEXITY_MODEL` (optional) — Perplexity model to call (defaults to `sonar`).
- `OPENAI_API_KEY`, `CHROMA_OPENAI_API_KEY` (optional) — only used if you change the code to call OpenAI embeddings.

Notes & best practices
----------------------
- Do not commit real API keys. Keep `.env` out of version control.
- The script uses local pseudo-embeddings by default for safe offline testing. Replace embedding logic if you need semantic OpenAI embeddings.
- The Perplexity client in `venv/app.py` logs request/response details for debugging; remove or reduce logging in production.

If you want, I can:
- add a `.env.example` to `venv/` instead of the repo root,
- add a `--mock-llm` mode to run fully offline, or
- switch to OpenAI embeddings (and update `requirements.txt`).
