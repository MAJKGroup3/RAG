import sys
import argparse
import hashlib
import mimetypes
import requests
from pathlib import Path
from bs4 import BeautifulSoup

import chromadb
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader


def extract_text_from_file(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime == "application/pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_text_from_url(url: str) -> str:
    resp = requests.get(url, headers={"User-Agent": "Legal-RAG/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    return soup.get_text(separator="\n")


def chunk_text(text: str, max_len=800, overlap=100):
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []

    for para in paras:
        if len(para) <= max_len:
            chunks.append(para)
            continue

        sentences = para.split(". ")
        curr, curr_len = [], 0
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if curr_len + len(s) <= max_len:
                curr.append(s)
                curr_len += len(s)
            else:
                if curr:
                    chunks.append(". ".join(curr))
                curr, curr_len = [s], len(s)
        if curr:
            chunks.append(". ".join(curr))

    final = []
    for i, c in enumerate(chunks):
        if i == 0:
            final.append(c)
        else:
            prev = final[-1]
            combined = (prev[-overlap:] + " " + c) if len(prev) > overlap else (prev + " " + c)
            final.append(combined)

    return final


def get_chroma(persist_dir: str):
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name="tos_docs")
    return client, collection


def ingest_sources(sources, db_dir="./chroma_legal"):
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    client, collection = get_chroma(persist_dir=db_dir)

    for src in sources:
        print(f"\n Ingesting: {src}")

        if src.startswith("http://") or src.startswith("https://"):
            text = extract_text_from_url(src)
            doc_id = hashlib.md5(src.encode()).hexdigest()
            meta = {"source": src, "type": "url"}
        else:
            path = Path(src)
            if not path.exists():
                print(f" File not found, skipping: {src}")
                continue
            text = extract_text_from_file(path)
            doc_id = hashlib.md5(path.read_bytes()).hexdigest()
            meta = {"source": str(path), "type": "file"}

        text = text.strip()
        if not text:
            print(" No text extracted, skipping.")
            continue

        chunks = chunk_text(text)
        if not chunks:
            print(" No chunks created, skipping.")
            continue

        embeddings = embed_model.encode(chunks).tolist()
        ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
        metadatas = [
            {**meta, "chunk_index": i}  # Add idx of chunk to metadata
            for i in range(len(chunks))
        ]

        print(f"   ✂ {len(chunks)} chunks -> storing into Chroma")
        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

    print("\n Persisting Chroma DB…")
    print("Ingestion complete.")


# ------------------------
# CLI
# ------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest legal docs (EULA/ToS/contracts) into Chroma."
    )
    parser.add_argument("inputs", nargs="+", help="Local files or URLs to ingest")
    parser.add_argument("--db", default="./chroma_legal", help="Chroma persistence directory")
    args = parser.parse_args()

    ingest_sources(args.inputs, db_dir=args.db)


if __name__ == "__main__":
    main()
