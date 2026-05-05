#!/usr/bin/env python3
"""
Basic RAG Implementation (Step 1)

Simple example without persistent storage.
Good for understanding the basics of RAG.
"""

from rag_system import KnowledgeBaseRAG
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def main():
    print("\n[Step 1] Basic RAG System\n")
    
    # Initialize
    print("Initializing...")
    llm = Ollama(model="llama2", base_url="http://localhost:11434")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Create RAG
    rag = KnowledgeBaseRAG(llm=llm, embed_model=embed_model)
    
    # Ingest
    urls = [
        "https://law.justia.com/codes/california/code-pen/part-1/",
        "https://law.justia.com/codes/california/code-pen/part-1/title-8/chapter-1/"
    ]
    rag.ingest_urls(urls)
    
    # Query
    print("\n" + "="*50)
    question = "What is the definition of murder?"
    print(f"Question: {question}")
    print("="*50)
    
    response = rag.query(question)
    print(f"\nAnswer: {response}")
    print()


if __name__ == "__main__":
    main()
