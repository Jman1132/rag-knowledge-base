#!/usr/bin/env python3
"""
RAG System with FastAPI REST Endpoints

Provides HTTP API for the RAG system.

Endpoints:
- POST /query - Query the knowledge base
- POST /ingest - Ingest new URLs
- GET /health - Health check
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from rag_system import PersistentKnowledgeBaseRAG
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import uvicorn


app = FastAPI(
    title="RAG Knowledge Base API",
    description="Retrieval Augmented Generation for California Penal Code",
    version="1.0.0"
)

# Initialize RAG system
llm = Ollama(model="llama2", base_url="http://localhost:11434")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
rag = PersistentKnowledgeBaseRAG(llm=llm, embed_model=embed_model)

# Initialize with default URLs
try:
    urls = [
        "https://law.justia.com/codes/california/code-pen/part-1/",
        "https://law.justia.com/codes/california/code-pen/part-1/title-8/chapter-1/"
    ]
    rag.ingest_urls(urls)
    print("✓ Knowledge base initialized")
except Exception as e:
    print(f"✗ Error initializing knowledge base: {e}")


class QueryRequest(BaseModel):
    """Query request model"""
    question: str
    similarity_top_k: int = 3


class QueryResponse(BaseModel):
    """Query response model"""
    answer: str
    sources: List[str]


class IngestRequest(BaseModel):
    """Ingest request model"""
    urls: List[str]
    force_refresh: bool = False


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "RAG Knowledge Base API"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the knowledge base"""
    try:
        result = rag.query(request.question, similarity_top_k=request.similarity_top_k)
        return QueryResponse(
            answer=result['answer'],
            sources=result['sources']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest(request: IngestRequest):
    """Ingest new URLs into the knowledge base"""
    try:
        rag.ingest_urls(request.urls, force_refresh=request.force_refresh)
        return {"status": "success", "message": f"Ingested {len(request.urls)} URL(s)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("\n" + "="*50)
    print("RAG Knowledge Base API")
    print("="*50)
    print("\nStarting server...")
    print("API Documentation: http://localhost:8000/docs")
    print("\n" + "="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
