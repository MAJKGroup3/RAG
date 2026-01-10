import sys
from typing import List
from sentence_transformers import SentenceTransformer
import chromadb
import boto3
import json


COLLECTION_NAME = "legal_eulas"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BEDROCK_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
TOP_K = 10


client = chromadb.PersistentClient(path="./chroma")
collection = client.get_collection(COLLECTION_NAME)
embedder = SentenceTransformer(EMBED_MODEL)



bedrock = boto3.client(
            "bedrock-runtime",
            region_name="us-west-2",
            aws_access_key_id="",
            aws_secret_access_key="",
            #aws_session_token=None,  # optional
        )



def build_prompt(query: str, chunks: List[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[{c['filename']} - chunk {c['chunk_index']}]\n{c['text']}"
        for c in chunks
    )

    return f"""You are a legal analysis assistant.
Use ONLY the provided clauses to answer the user's query.
Use bullet points when neccessary and do not hesitate to provide more context when needed.
If not present, say you cannot find it in the provided text.

User query:
{query}

Relevant clauses:
{context}

Answer:
"""


def build_rewrite_prompt(query: str) -> str:
    return f"""Rewrite the user query for semantic search over legal agreements.
Preserve intent and key legal terms.
Return ONLY the rewritten query text.

User query:
{query}

Rewritten query:
"""


def call_bedrock(prompt: str, max_tokens: int = 1024, temperature: float = 0.0):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    response = bedrock.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


def rewrite_query(query: str) -> str:
    rewritten = call_bedrock(build_rewrite_prompt(query), max_tokens=128, temperature=0.2)
    return rewritten.strip().strip('"').strip()


def run_query(query: str):
    rewritten_query = rewrite_query(query)
    query_vec = embedder.encode([rewritten_query]).tolist()

    results = collection.query(
        query_embeddings=query_vec,
        n_results=TOP_K,
        include=["documents", "metadatas"]
    )

    chunks = []
    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "text": text,
            "filename": meta["filename"],
            "chunk_index": meta["chunk_index"],
            "source": meta.get("source", "")
        })

    prompt = build_prompt(query, chunks)
    answer = call_bedrock(prompt, max_tokens=1024, temperature=0.0)

    print("Answer:")
    print(answer)
    print("\nSources:")
    for c in chunks:
        print(f"- {c['filename']} (chunk {c['chunk_index']})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query.py \"your question here\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    run_query(question)
