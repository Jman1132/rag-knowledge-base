#!/usr/bin/env python3
"""
Dedicated RAG System for California Penal Code

Pulls from:
- https://law.justia.com/codes/california/code-pen/part-1/
- https://law.justia.com/codes/california/code-pen/part-1/title-8/chapter-1/

Optimized for legal code queries with proper formatting and citations.
"""

from rag_system import PersistentKnowledgeBaseRAG
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import sys


class CaliforniaPenalCodeRAG(PersistentKnowledgeBaseRAG):
    """Specialized RAG for California Penal Code"""
    
    def __init__(self, llm, embed_model, storage_dir="./storage/penal_code"):
        super().__init__(llm, embed_model, storage_dir)
        self.penal_code_urls = [
            "https://law.justia.com/codes/california/code-pen/part-1/",
            "https://law.justia.com/codes/california/code-pen/part-1/title-8/chapter-1/"
        ]
    
    def setup(self, force_refresh=False):
        """Setup and ingest California Penal Code"""
        print("\nSetting up California Penal Code Knowledge Base...")
        self.ingest_urls(self.penal_code_urls, force_refresh=force_refresh)
        print("✓ California Penal Code loaded successfully\n")
    
    def legal_query(self, question):
        """Query with legal formatting"""
        result = self.query(question)
        return self._format_legal_response(result)
    
    def _format_legal_response(self, result):
        """Format response with legal citations"""
        formatted = {
            "question": result.get("answer"),
            "sources": result.get("sources", [])
        }
        return formatted


def main():
    print("\n" + "="*70)
    print(" "*15 + "CALIFORNIA PENAL CODE Q&A")
    print(" "*10 + "Powered by Retrieval Augmented Generation")
    print("="*70 + "\n")
    
    # Initialize
    print("[1/2] Initializing AI models...")
    try:
        llm = Ollama(model="llama2", base_url="http://localhost:11434")
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        print("✓ Models ready\n")
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nEnsure Ollama is running: ollama serve")
        sys.exit(1)
    
    print("[2/2] Loading California Penal Code...")
    rag = CaliforniaPenalCodeRAG(llm=llm, embed_model=embed_model)
    
    try:
        rag.setup(force_refresh=False)
    except Exception as e:
        print(f"✗ Error loading knowledge base: {e}")
        sys.exit(1)
    
    # Display help
    print("="*70)
    print("KNOWLEDGE BASE READY")
    print("="*70)
    print("\nPart 1: Of Crimes and Punishments (§25-680.4)")
    print("  - All titles and chapters of California Penal Code Part 1")
    print("\nTitle 8: Of Crimes Against the Person (§187-248)")
    print("  - Chapter 1: Homicide (§187-199)")
    print("\n" + "-"*70)
    print("\nExample Questions:")
    print("  1. What is the definition of murder under California law?")
    print("  2. What are the degrees of murder?")
    print("  3. What is the penalty for first-degree murder?")
    print("  4. What is the difference between murder and manslaughter?")
    print("  5. Explain express and implied malice.")
    print("  6. What constitutes voluntary manslaughter?")
    print("  7. What is involuntary manslaughter?")
    print("\n" + "-"*70)
    print("\nCommands: Type 'quit' or 'exit' to exit\n")
    print("="*70 + "\n")
    
    # Q&A Loop
    while True:
        try:
            question = input("\nQuestion: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using California Penal Code Q&A!\n")
                break
            
            if not question:
                continue
            
            print("\n[Searching...]")
            result = rag.query(question)
            
            print(f"\nAnswer:")
            print("-" * 70)
            print(result['answer'])
            print("-" * 70)
            
            if result['sources']:
                print(f"\nLegal Sources:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  [{i}] {source}")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!\n")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    main()
