import argparse
import email
import hashlib
import imaplib
import os
import time
from pathlib import Path
from typing import Any, Iterable

import chromadb
import requests
from dotenv import load_dotenv

# Load environment variables from venv/.env (relative to this file)
_dotenv_path = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=_dotenv_path)

DEBUG = os.getenv("DEBUG", "0") == "1" or os.getenv("PERPLEXITY_DEBUG", "0") == "1"

def _redact_bearer(token: str) -> str:
    if not token:
        return ""
    if len(token) <= 10:
        return "***"
    return token[:7] + "…" + token[-4:]

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")


def _http_error_message(response: requests.Response) -> str:
    try:
        data = response.json()
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, str) and err.strip():
                return err
    except Exception:
        pass
    return response.text.strip()[:500]


def get_embedding(text: str) -> list[float]:
    """Get a semantic embedding from Ollama.

    Uses OLLAMA_EMBED_MODEL via POST {OLLAMA_HOST}/api/embeddings.
    """
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
            timeout=60,
        )
    except requests.RequestException as e:
        raise RuntimeError(
            "Failed to reach Ollama embeddings endpoint. "
            "Ensure Ollama is running and reachable at OLLAMA_HOST (default http://localhost:11434). "
            f"Original error: {e}"
        ) from e

    if not response.ok:
        msg = _http_error_message(response)
        raise RuntimeError(
            f"Ollama embeddings request failed ({response.status_code}). "
            f"Model={OLLAMA_EMBED_MODEL}. Details: {msg}"
        )

    data = response.json()
    embedding = data.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise RuntimeError("Ollama embeddings response missing 'embedding' array.")
    return embedding


def get_chroma_collection() -> Any:
    chroma_client = chromadb.PersistentClient(path="./chroma_persistent_storage")

    # Best practice: keep separate collections per embedding model/dimension.
    # This avoids dimension-mismatch errors when switching embedding backends.
    collection_name = os.getenv(
        "CHROMA_COLLECTION",
        f"rag_{_sanitize_collection_suffix(OLLAMA_EMBED_MODEL)}",
    )
    return chroma_client.get_or_create_collection(name=collection_name)


def _sanitize_collection_suffix(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return "default"
    return "".join((c if c.isalnum() else "_") for c in value)

class OllamaChatClient:
    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model

    def chat(self, prompt: str, system: str = "Be precise and concise.") -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }

        if DEBUG:
            print("==== Ollama chat request ====")
            print({"url": f"{self.host}/api/chat", "payload": payload})

        try:
            response = requests.post(f"{self.host}/api/chat", json=payload, timeout=120)
        except requests.RequestException as e:
            raise RuntimeError(
                "Failed to reach Ollama chat endpoint. "
                "Ensure Ollama is running and reachable at OLLAMA_HOST (default http://localhost:11434). "
                f"Original error: {e}"
            ) from e

        if DEBUG:
            print("==== Ollama chat response status ====", response.status_code)
            print("==== Ollama chat response body ====")
            print(response.text)

        if not response.ok:
            msg = _http_error_message(response)
            raise RuntimeError(
                f"Ollama chat request failed ({response.status_code}). "
                f"Model={self.model}. Details: {msg}"
            )

        data = response.json()
        message = data.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
        raise RuntimeError("Ollama chat response missing message.content.")


ollama_client = OllamaChatClient(host=OLLAMA_HOST, model=OLLAMA_CHAT_MODEL)


def load_documents_from_directory(directory_path: str = "./news_articles") -> list[dict[str, str]]:
    print("==== Loading documents from directory ====")
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        print(f"Created directory {directory_path}. Add .txt files there.")
        return []

    documents: list[dict[str, str]] = []
    for filename in sorted(os.listdir(directory_path)):
        if not filename.endswith(".txt"):
            continue
        full_path = os.path.join(directory_path, filename)
        with open(full_path, "r", encoding="utf-8") as file:
            documents.append({"id": filename, "text": file.read(), "source": full_path})
    return documents


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 20) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _state_file_for_collection() -> Path:
    name = os.getenv(
        "CHROMA_COLLECTION",
        f"rag_{_sanitize_collection_suffix(OLLAMA_EMBED_MODEL)}",
    )
    return Path("./chroma_persistent_storage") / f"index_state_{name}.txt"


def _load_index_state() -> dict[str, str]:
    state_path = _state_file_for_collection()
    if not state_path.exists():
        return {}
    state: dict[str, str] = {}
    for line in state_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "\t" not in line:
            continue
        source, digest = line.split("\t", 1)
        state[source] = digest
    return state


def _save_index_state(state: dict[str, str]) -> None:
    state_path = _state_file_for_collection()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}\t{v}" for k, v in sorted(state.items())]
    state_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def index_directory(
    directory_path: str = "./news_articles",
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 20,
    incremental: bool = True,
) -> int:
    collection = get_chroma_collection()
    documents = load_documents_from_directory(directory_path)
    print(f"Loaded {len(documents)} documents")

    state = _load_index_state() if incremental else {}
    updated_state: dict[str, str] = dict(state)

    total_chunks_upserted = 0
    for doc in documents:
        source = doc.get("source", doc["id"])
        digest = _file_sha256(source) if os.path.isfile(source) else hashlib.sha256(doc["text"].encode("utf-8")).hexdigest()
        if incremental and state.get(source) == digest:
            continue

        chunks = split_text(doc["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print("==== Splitting docs into chunks ====")

        for i, chunk in enumerate(chunks, start=1):
            chunk_id = f"{doc['id']}_chunk{i}"
            print("==== Generating embeddings... ====")
            embedding = get_embedding(chunk)
            print("==== Inserting chunks into db ====")
            collection.upsert(
                ids=[chunk_id],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"source": source, "chunk": i, "doc_id": doc["id"]}],
            )
            total_chunks_upserted += 1

        updated_state[source] = digest

    if incremental:
        _save_index_state(updated_state)

    return total_chunks_upserted


def query_documents(question: str, n_results: int = 2) -> list[str]:
    collection = get_chroma_collection()
    q_emb = get_embedding(question)
    results = collection.query(query_embeddings=[q_emb], n_results=n_results)
    relevant_chunks = [doc for sublist in results.get("documents", []) for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks


def query_documents_with_sources(question: str, n_results: int = 2) -> list[dict[str, Any]]:
    collection = get_chroma_collection()
    q_emb = get_embedding(question)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    ids = (results.get("ids") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]
    out: list[dict[str, Any]] = []
    for i in range(min(len(docs), len(metas), len(ids), len(dists))):
        out.append({"id": ids[i], "text": docs[i], "metadata": metas[i], "distance": dists[i]})
    print("==== Returning relevant chunks (with sources) ====")
    return out


def generate_response(question: str, relevant_chunks: list[str]) -> str:
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )
    return ollama_client.chat(f"{prompt}\nAnswer:")


def generate_response_with_sources(question: str, retrieved: list[dict[str, Any]]) -> str:
    chunks = [r.get("text", "") for r in retrieved]
    answer = generate_response(question, chunks)
    sources: list[str] = []
    for r in retrieved:
        meta = r.get("metadata") or {}
        source = meta.get("source")
        chunk = meta.get("chunk")
        if source:
            sources.append(f"{source}#chunk{chunk}" if chunk else str(source))
    if sources:
        answer = answer.rstrip() + "\n\nSources:\n" + "\n".join(sorted(set(sources)))
    return answer


def _iter_txt_files(directory: str) -> Iterable[str]:
    p = Path(directory)
    if not p.exists():
        return []
    return (str(x) for x in sorted(p.glob("*.txt")))


def watch_directory(directory: str, poll_seconds: int = 10) -> None:
    print(f"Watching directory for .txt changes: {directory}")
    while True:
        try:
            upserted = index_directory(directory, incremental=True)
            if upserted:
                print(f"Indexed {upserted} new/updated chunks")
        except Exception as e:
            print(f"[watch_directory] Error: {e}")
        time.sleep(max(1, poll_seconds))


def ingest_from_imap(
    *,
    out_dir: str,
    host: str,
    user: str,
    password: str,
    mailbox: str = "INBOX",
    subject_contains: str | None = None,
    mark_seen: bool = True,
) -> int:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    imap = imaplib.IMAP4_SSL(host)
    imap.login(user, password)
    imap.select(mailbox)

    criteria: list[str] = ["UNSEEN"]
    if subject_contains:
        criteria.extend(["SUBJECT", f'"{subject_contains}"'])

    typ, data = imap.search(None, *criteria)
    if typ != "OK":
        raise RuntimeError(f"IMAP search failed: {typ}")

    msg_ids = [x for x in (data[0] or b"").split() if x]
    saved = 0
    for msg_id in msg_ids:
        typ, msg_data = imap.fetch(msg_id, "(RFC822)")
        if typ != "OK" or not msg_data:
            continue
        raw = msg_data[0][1]
        msg = email.message_from_bytes(raw)

        subject = (msg.get("Subject") or "").strip()
        date_hdr = (msg.get("Date") or "").strip()
        base_name = f"email_{msg_id.decode(errors='ignore')}_{int(time.time())}"

        extracted_any = False

        if msg.is_multipart():
            for part in msg.walk():
                ctype = (part.get_content_type() or "").lower()
                disp = (part.get("Content-Disposition") or "").lower()
                filename = part.get_filename()

                if filename and filename.lower().endswith(".txt"):
                    payload = part.get_payload(decode=True) or b""
                    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in filename)
                    target = out_path / safe_name
                    target.write_bytes(payload)
                    saved += 1
                    extracted_any = True
                    continue

                if "attachment" not in disp and ctype == "text/plain" and not extracted_any:
                    payload = part.get_payload(decode=True) or b""
                    text = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                    if text.strip():
                        target = out_path / f"{base_name}.txt"
                        target.write_text(
                            f"Subject: {subject}\nDate: {date_hdr}\n\n{text}",
                            encoding="utf-8",
                        )
                        saved += 1
                        extracted_any = True
        else:
            ctype = (msg.get_content_type() or "").lower()
            if ctype == "text/plain":
                payload = msg.get_payload(decode=True) or b""
                text = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
                if text.strip():
                    target = out_path / f"{base_name}.txt"
                    target.write_text(
                        f"Subject: {subject}\nDate: {date_hdr}\n\n{text}",
                        encoding="utf-8",
                    )
                    saved += 1
                    extracted_any = True

        if mark_seen and extracted_any:
            imap.store(msg_id, "+FLAGS", "\\Seen")

    imap.logout()
    return saved


def watch_imap_and_index() -> None:
    host = os.getenv("EMAIL_IMAP_HOST", "").strip()
    user = os.getenv("EMAIL_IMAP_USER", "").strip()
    password = os.getenv("EMAIL_IMAP_PASSWORD", "").strip()
    mailbox = os.getenv("EMAIL_IMAP_MAILBOX", "INBOX").strip() or "INBOX"
    subject = os.getenv("EMAIL_SUBJECT_FILTER", "").strip() or None
    out_dir = os.getenv("EMAIL_OUT_DIR", "./news_articles/email_inbox").strip() or "./news_articles/email_inbox"
    poll_seconds = int(os.getenv("EMAIL_POLL_SECONDS", "30"))

    if not host or not user or not password:
        raise RuntimeError("EMAIL_IMAP_HOST/EMAIL_IMAP_USER/EMAIL_IMAP_PASSWORD must be set to use --watch-email")

    print(f"Watching IMAP inbox: host={host} user={user} mailbox={mailbox} out_dir={out_dir}")
    while True:
        try:
            saved = ingest_from_imap(
                out_dir=out_dir,
                host=host,
                user=user,
                password=password,
                mailbox=mailbox,
                subject_contains=subject,
                mark_seen=True,
            )
            if saved:
                print(f"Saved {saved} email-derived .txt files; indexing...")
                upserted = index_directory("./news_articles", incremental=True)
                print(f"Indexed {upserted} new/updated chunks")
        except Exception as e:
            print(f"[watch_email] Error: {e}")
        time.sleep(max(5, poll_seconds))


def _run_default_demo() -> None:
    index_directory("./news_articles", incremental=True)
    print("\n=== Ollama standalone test ===")
    print(ollama_client.chat("Explain the theory of relativity in simple terms."))

    question = "tell me about databricks"
    print("\n=== RAG Query: " + question + " ===")
    retrieved = query_documents_with_sources(question, n_results=2)
    answer = generate_response_with_sources(question, retrieved)
    print(answer)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local RAG demo (Ollama + Chroma)")
    parser.add_argument("--index", action="store_true", help="Index ./news_articles and exit")
    parser.add_argument("--watch-dir", metavar="DIR", help="Watch a directory for new/changed .txt files and index incrementally")
    parser.add_argument("--watch-email", action="store_true", help="Poll IMAP for unseen mail, save .txt locally, then index")
    parser.add_argument("--ask", metavar="QUESTION", help="Ask a question using the current Chroma index")
    parser.add_argument("--n-results", type=int, default=2, help="Number of retrieved chunks")
    args = parser.parse_args(argv)

    if args.watch_dir:
        watch_directory(args.watch_dir, poll_seconds=int(os.getenv("WATCH_POLL_SECONDS", "10")))
        return 0

    if args.watch_email:
        watch_imap_and_index()
        return 0

    if args.index:
        upserted = index_directory("./news_articles", incremental=True)
        print(f"Indexed {upserted} new/updated chunks")
        return 0

    if args.ask:
        retrieved = query_documents_with_sources(args.ask, n_results=args.n_results)
        print(generate_response_with_sources(args.ask, retrieved))
        return 0

    _run_default_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
