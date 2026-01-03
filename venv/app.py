import os
from pathlib import Path
import chromadb
import requests
from dotenv import load_dotenv

# Load environment variables from venv/.env (relative to this file)
_dotenv_path = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=_dotenv_path)

# Optional debug logging (off by default)
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


# Persistent ChromaDB client + collection
chroma_client = chromadb.PersistentClient(path="./chroma_persistent_storage")


def _sanitize_collection_suffix(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return "default"
    return "".join((c if c.isalnum() else "_") for c in value)


# Best practice: keep separate collections per embedding model/dimension.
# This avoids dimension-mismatch errors when switching embedding backends.
collection_name = os.getenv(
    "CHROMA_COLLECTION",
    f"rag_{_sanitize_collection_suffix(OLLAMA_EMBED_MODEL)}",
)
collection = chroma_client.get_or_create_collection(name=collection_name)

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


def load_documents_from_directory(directory_path: str = "./news_articles") -> list[dict]:
    print("==== Loading documents from directory ====")
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        print(f"Created directory {directory_path}. Add .txt files there.")
        return []

    documents: list[dict] = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as file:
                documents.append({"id": filename, "text": file.read()})
    return documents


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 20) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


documents = load_documents_from_directory("./news_articles")
print(f"Loaded {len(documents)} documents")

chunked_documents: list[dict] = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})


for doc in chunked_documents:
    print("==== Generating embeddings... ====")
    doc["embedding"] = get_embedding(doc["text"])

for doc in chunked_documents:
    print("==== Inserting chunks into db ====")
    collection.upsert(
        ids=[doc["id"]],
        documents=[doc["text"]],
        embeddings=[doc["embedding"]],
    )


def query_documents(question: str, n_results: int = 2) -> list[str]:
    q_emb = get_embedding(question)
    results = collection.query(query_embeddings=[q_emb], n_results=n_results)
    relevant_chunks = [doc for sublist in results.get("documents", []) for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks


def generate_response(question: str, relevant_chunks: list[str]) -> str:
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )
    return ollama_client.chat(f"{prompt}\nAnswer:")


print("\n=== Ollama standalone test ===")
print(ollama_client.chat("Explain the theory of relativity in simple terms."))

question = "tell me about databricks"
print("\n=== RAG Query: " + question + " ===")
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)
print(answer)
