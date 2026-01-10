import sys
import json
import argparse
from typing import List
from sentence_transformers import SentenceTransformer
import chromadb
import boto3

# CONFIG
COLLECTION_NAME = "tos_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
TOP_K = 10

# LOAD SERVICES
client = chromadb.PersistentClient(path="./chroma_legal")
collection = client.get_collection(COLLECTION_NAME)
embedder = SentenceTransformer(EMBED_MODEL)

bedrock = boto3.client(
    "bedrock-runtime",
    region_name="us-west-2",
    aws_access_key_id="xxx",
    aws_secret_access_key="xxx",
)


def build_prompt(query: str, chunks: List[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[{c['source']} - chunk {c['chunk_index']}]\n{c['text']}"
        for c in chunks
    )

    return f"""You are a legal analysis assistant.
Use ONLY the provided clauses to answer the user's query.
If not present, say you cannot find it in the provided text.

User query:
{query}

Relevant clauses:
{context}

Answer:
"""


def call_bedrock(prompt: str):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.0,
    }

    response = bedrock.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


def run_query(query: str, source_filter: str = None):
    query_vec = embedder.encode([query]).tolist()

    query_kwargs = {
        "query_embeddings": query_vec,
        "n_results": TOP_K,
        "include": ["documents", "metadatas"]
    }

    if source_filter:
        query_kwargs["where"] = {"source": {"$eq": source_filter}}

    results = collection.query(**query_kwargs)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        print("No relevant information found for this source/query.")
        return

    chunks = [
        {
            "text": t,
            "source": m.get("source", "unknown"),
            "chunk_index": m.get("chunk_index", i),
        }
        for i, (t, m) in enumerate(zip(docs, metas))
    ]

    prompt = build_prompt(query, chunks)
    answer = call_bedrock(prompt)

    print("\n Answer:")
    print(answer)
    print("\n Sources:")
    for c in chunks:
        print(f"- {c['source']} (chunk {c['chunk_index']})")

# ------------------------
# CLI
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query ingested legal documents via Bedrock")
    parser.add_argument("question", type=str, help="The question to ask")
    parser.add_argument("--source", type=str, default=None, help="Optional source URL or filename filter")
    args = parser.parse_args()

    run_query(args.question, source_filter=args.source)
