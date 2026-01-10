#!/usr/bin/env python3
"""
Eula RAG - Local Version
Just fetch data, create embeddings, store in ChromaDB, query locally
"""

import json
import boto3
import os
import hashlib
from langchain_aws import BedrockEmbeddings
from chromadb import CloudClient
import config
#from pypdf import PdfReader
import pdfplumber

s3_client = boto3.client("s3", region_name=config.AWS_REGION)

embeddings_model = BedrockEmbeddings(
    model_id=config.EMBEDDING_MODEL,
    region_name=config.AWS_REGION
)

bedrock_runtime = boto3.client("bedrock-runtime", region_name=config.AWS_REGION)

chroma_client = CloudClient(
    api_key=config.CHROMA_API_KEY,
    tenant=config.CHROMA_TENANT,
    database=config.CHROMA_DATABASE
)



def extract_pdf(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append({
                    "text": text,
                    "page": i + 1
                })
    return pages


def save_documents_to_s3(local_pdf_dir = "eulas", s3_prefix = "documents/"):

    for filename in os.listdir(local_pdf_dir):
        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(local_pdf_dir, filename)
        print(f"Extracting {filename}")

        pages = extract_pdf(pdf_path)

        full_text = "\n\n".join(
            f"[PAGE {p['page']}]\n{p['text']}"
            for p in pages
        )

        txt_name = filename.replace(".pdf", ".txt")
        s3_key = s3_prefix + txt_name

        print(f"Uploading {txt_name} → s3://{config.S3_BUCKET_NAME}/{s3_key}")

        s3_client.put_object(
            Bucket = config.S3_BUCKET_NAME,
            Key = s3_key,
            Body = full_text.encode("utf-8"),
            ContentType = "text/plain"
        )


def recursive_chunk(text, max_chars = 1800, overlap = 200):
    separators = [
        "\nSECTION ",
        "\nArticle ",
        "\n\n",
        "\n",
        ". "
    ]

    chunks = [text.strip()]

    for sep in separators:
        next_chunks = []
        for c in chunks:
            if len(c) <= max_chars:
                next_chunks.append(c)
                continue

            parts = c.split(sep)
            buf = ""

            for p in parts:
                add = p + sep if sep != " " else p + " "
                if len(buf) + len(add) <= max_chars:
                    buf += add
                else:
                    if buf.strip():
                        next_chunks.append(buf.strip())
                    buf = add

            if buf.strip():
                next_chunks.append(buf.strip())

        chunks = next_chunks

    final = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            step = max(1, max_chars - overlap)
            for i in range(0, len(c), step):
                piece = c[i : i + max_chars].strip()
                if piece:
                    final.append(piece)

    return final


def ingest_from_s3():
    bucket = config.S3_BUCKET_NAME
    prefix = "documents/"

    os.makedirs("documents", exist_ok = True)

    collection = chroma_client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME
    )

    objects = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix).get(
        "Contents", []
    )

    for obj in objects:
        key = obj["Key"]
        if not key.lower().endswith(".txt"):
            continue

        print(f"Ingesting {key}")

        local_path = os.path.join("documents", os.path.basename(key))
        s3_client.download_file(bucket, key, local_path)

        with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        chunks = recursive_chunk(text)

        ids = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            cid = hashlib.sha256(f"{key}:{i}".encode("utf-8")).hexdigest()[:32]
            ids.append(cid)
            metadatas.append(
                {
                    "document": key,
                    "chunk_index": i,
                    "doc_type": "EULA",
#                  "effective_date": "2024-10-01"
                }
            )

        embeddings = embeddings_model.embed_documents(chunks)

        if hasattr(collection, "upsert"):
            collection.upsert(
                ids=ids,
                documents=chunks,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        else:
            collection.add(
                ids=ids,
                documents=chunks,
                metadatas=metadatas,
                embeddings=embeddings,
            )


def search(query, top_k=5):
    print(f"\nSearching for: '{query}'")

    collection = chroma_client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME
    )

    query_embedding = embeddings_model.embed_query(query) 

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    formatted = []
    for i in range(len(results["ids"][0])):
        formatted.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return formatted


def generate_answer(query, results):
    context = "\n\n".join(
        [f"[{r['metadata']['document']}]\n{r['text']}" for r in results[:3]]
    )

    prompt = f"""
You are a legal document assistant.
Answer the question using ONLY the provided context.
If the answer is not explicitly stated, say:
"The documents provided do not specify this." and nothing else.

Do not provide legal advice.

Context:
{context}

Question:
{query}
"""

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 500,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }

    response = bedrock_runtime.invoke_model(
        modelId = config.LLM_MODEL, 
        body = json.dumps(request_body)
    )

    response_body = json.loads(response["body"].read())
    return response_body["content"][0]["text"]


def main():
    import sys

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python eula_rag.py build")
        print("  python eula_rag.py ingest")
        print("  python eula_rag.py query 'question'")
        return

    command = sys.argv[1]

    if command == "build":
        save_documents_to_s3()
        print("Raw documents saved to S3")
    elif command == "ingest":
        ingest_from_s3()
        print("S3 documents ingested into Chroma")
    elif command == "query":
        query_text = " ".join(sys.argv[2:])
        results = search(query_text)
        answer = generate_answer(query_text, results)
        print(answer)

    else:
        print("Use: build, ingest, or query")


if __name__ == "__main__":
    main()