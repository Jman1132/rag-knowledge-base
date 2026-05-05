#!/usr/bin/env python3
"""
Advanced RAG System with Persistent Storage and Interactive Q&A

This script demonstrates Step 3 of the RAG implementation:
- Persistent storage of indexed documents
- Interactive Q&A loop
- Source citations
- Caching for performance

Knowledge Base: California Penal Code
- Part 1: Of Crimes and Punishments (§25-680.4)
- Title 8, Chapter 1: Homicide (§187-199)
"""

from rag_system import PersistentKnowledgeBaseRAG
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import sys


def main():
    print("\n" + "="*60)
    print("RAG Knowledge Base Q&A System")
    print("California Penal Code Knowledge Base")
    print("="*60 + "\n")
    
    # Initialize LLM and embedding model
    print("[1/3] Initializing models...")
    try:
        llm = Ollama(model="llama2", base_url="http://localhost:11434")
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        print("✓ Models initialized successfully\n")
    except Exception as e:
        print(f"✗ Error initializing models: {e}")
        print("\nMake sure Ollama is running: ollama serve")
        sys.exit(1)
    
    # Initialize RAG system
    print("[2/3] Setting up RAG system...")
    rag = PersistentKnowledgeBaseRAG(llm=llm, embed_model=embed_model)
    print("✓ RAG system initialized\n")
    
    # California Penal Code URLs
    urls = [
        "https://law.justia.com/codes/california/code-pen/part-1/",
        "https://law.justia.com/codes/california/code-pen/part-1/title-8/chapter-1/"
    ]
    
    # Ingest documents
    print("[3/3] Ingesting knowledge base...")
    print(f"Loading {len(urls)} URL(s)...\n")
    
    try:
        rag.ingest_urls(urls)
    except Exception as e:
        print(f"✗ Error ingesting URLs: {e}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Knowledge Base Ready!")
    print("="*60)
    print("\nExample questions you can ask:")
    print("  - What is the definition of murder in California?")
    print("  - What are the degrees of murder?")
    print("  - What is the penalty for first-degree murder?")
    print("  - Explain voluntary vs involuntary manslaughter")
    print("  - What constitutes implied malice?\n")
    print("Type 'quit' or 'exit' to stop\n")
    print("="*60 + "\n")
    
    # Interactive Q&A loop
    while True:
        try:
            question = input("\nYou: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using RAG Knowledge Base!")
                break
            
            if not question:
                continue
            
            print("\n[Searching knowledge base...]")
            result = rag.query(question)
            
            print(f"\nAssistant: {result['answer']}")
            
            if result['sources']:
                print(f"\nSources:")
                for source in result['sources']:
                    print(f"  • {source}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error processing query: {e}")
            continue


if __name__ == "__main__":
    main()
