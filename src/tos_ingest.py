import os
import sys
import time
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


# CONFIG
SOURCE_DIR = "legal_docs"
COLLECTION_NAME = "legal_eulas"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# HELPERS: FILE READERS
def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_txt(path: Path) -> str:
    return path.read_text(errors="ignore")



# INGEST TEXT FLOW
def recursive_chunk(text: str, max_len=600, overlap=80) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+max_len]
        chunks.append(" ".join(chunk))
        i += max_len - overlap
    return chunks


def infer_doc_type(name: str) -> str:
    name = name.lower()
    if "tos" in name or "terms" in name: return "ToS"
    if "eula" in name: return "EULA"
    if "contract" in name: return "Contract"
    return "Unknown"


# INGEST SINGLE SOURCE
def ingest_source(src: str, collection, embedder):
    path = Path(src)
    if path.is_dir():
        for p in path.glob("*"):
            ingest_source(str(p), collection, embedder)
        return

    if not path.exists():
        raise FileNotFoundError(f"Not a file or url: {src}")

    ext = path.suffix.lower()
    if ext == ".pdf": text = load_pdf(path)
    elif ext == ".txt": text = load_txt(path)
    else: raise ValueError(f"Unsupported file extension: {ext}")

    filename = path.name

    print(f"Processing {filename}")
    chunks = recursive_chunk(text)
    print(f"   ✂ {len(chunks)} chunks")

    embeddings = embedder.encode(chunks).tolist()

    ids = [f"{filename}-{i}" for i in range(len(chunks))]
    metadatas = [{
        "filename": filename,
        "chunk_index": i,
        "doc_type": infer_doc_type(filename),
        "source": src
    } for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )

    time.sleep(0.2)


# ------------------------
# MAIN INGEST LOGIC
# ------------------------
def ingest_cli(inputs=None):
    api_key=''
    tenant=''
    database=''

    client = chromadb.CloudClient(
        api_key=api_key,
        tenant=tenant,
        database=database
    )

    embedder = SentenceTransformer(EMBED_MODEL)
    COLLECTION_NAME = "rag_storage"
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    if inputs:
        for src in inputs:
            ingest_source(src, collection, embedder)
    else:
        # default directory behavior
        print("Loading documents from directory:", SOURCE_DIR)
        for path in Path(SOURCE_DIR).glob("*"):
            ingest_source(str(path), collection, embedder)

    print("Persisting Chroma...")
    print("Done.")


if __name__ == "__main__":
    args = sys.argv[1:]
    ingest_cli(args if args else None)
