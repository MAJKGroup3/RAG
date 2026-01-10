"""
Simple Configuration
"""

# AWS Configuration
AWS_REGION = 'us-east-1'
S3_BUCKET_NAME = 'PUT YOUR BUCKET NAME HERE'

# Bedrock Configuration
EMBEDDING_MODEL = 'amazon.titan-embed-text-v1'
LLM_MODEL = 'anthropic.claude-3-5-sonnet-20240620-v1:0'

# ChromaDB Cloud Configuration
CHROMA_API_KEY = "PUT YOUR API KEY HERE"
CHROMA_TENANT = "default_tenant" 
CHROMA_DATABASE = "PUT YOUR DATABASE NAME HERE" 
CHROMA_COLLECTION_NAME = "PUT YOUR COLLECTION NAME HERE"