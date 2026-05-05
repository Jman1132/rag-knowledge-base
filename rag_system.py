#!/usr/bin/env python3
"""
Core RAG (Retrieval Augmented Generation) System

Provides three implementations:
1. KnowledgeBaseRAG - Basic RAG without persistence
2. PersistentKnowledgeBaseRAG - Advanced with persistent storage (Step 3)
3. EnhancedRAG - Chat and batch features
"""

from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
from llama_index.readers.web import WebPageReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from pathlib import Path
from typing import List, Dict, Optional


class KnowledgeBaseRAG:
    """Basic RAG System (Step 1)"""
    
    def __init__(self, llm, embed_model):
        self.llm = llm
        self.embed_model = embed_model
        self.index = None
    
    def ingest_urls(self, urls: List[str]):
        """Fetch and index web pages from URLs"""
        print(f"Fetching {len(urls)} URLs...")
        reader = WebPageReader()
        documents = reader.load_data(urls=urls)
        
        print(f"Indexing {len(documents)} documents...")
        from llama_index.core import Settings
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        
        self.index = VectorStoreIndex.from_documents(documents)
        print("✓ Index created successfully!")
    
    def query(self, question: str) -> str:
        """Query the knowledge base"""
        if self.index is None:
            raise ValueError("No index created. Call ingest_urls() first.")
        
        query_engine = self.index.as_query_engine()
        response = query_engine.query(question)
        return response


class PersistentKnowledgeBaseRAG(KnowledgeBaseRAG):
    """Advanced RAG with Persistent Storage (Step 3)"""
    
    def __init__(self, llm, embed_model, storage_dir: str = "./storage"):
        super().__init__(llm, embed_model)
        self.storage_dir = storage_dir
        self._setup_storage()
    
    def _setup_storage(self):
        """Setup persistent storage directory"""
        Path(self.storage_dir).mkdir(exist_ok=True)
    
    def ingest_urls(self, urls: List[str], force_refresh: bool = False):
        """Fetch and index web pages with persistent storage"""
        storage_path = Path(self.storage_dir) / "index_store"
        
        # Load existing index if available
        if storage_path.exists() and not force_refresh:
            print("✓ Loading existing index from storage...")
            storage_context = StorageContext.from_defaults(persist_dir=str(storage_path))
            self.index = load_index_from_storage(storage_context)
            return
        
        print(f"Fetching {len(urls)} URLs...")
        reader = WebPageReader()
        documents = reader.load_data(urls=urls)
        
        print(f"Indexing {len(documents)} documents...")
        from llama_index.core import Settings
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        
        self.index = VectorStoreIndex.from_documents(documents)
        
        # Persist index
        self.index.storage_context.persist(persist_dir=str(storage_path))
        print(f"✓ Index persisted to {storage_path}")
    
    def query(self, question: str, similarity_top_k: int = 3) -> Dict:
        """Query with source citations"""
        if self.index is None:
            raise ValueError("No index created. Call ingest_urls() first.")
        
        query_engine = self.index.as_query_engine(similarity_top_k=similarity_top_k)
        response = query_engine.query(question)
        
        return {
            "answer": str(response),
            "sources": self._extract_sources(response)
        }
    
    def _extract_sources(self, response) -> List[str]:
        """Extract source URLs from response"""
        sources = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                if 'source' in node.metadata:
                    sources.append(node.metadata['source'])
        return list(set(sources))


class EnhancedRAG(PersistentKnowledgeBaseRAG):
    """Enhanced RAG with Chat and Batch Features (Step 4)"""
    
    def __init__(self, llm, embed_model, storage_dir: str = "./storage"):
        super().__init__(llm, embed_model, storage_dir)
        self.conversation_history = []
    
    def ingest_urls_with_chunking(self, urls: List[str], chunk_size: int = 512, chunk_overlap: int = 20):
        """Ingest with custom chunking"""
        print(f"Fetching {len(urls)} URLs...")
        reader = WebPageReader()
        documents = reader.load_data(urls=urls)
        
        # Custom node parser for better chunking
        parser = SimpleNodeParser.from_defaults(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        from llama_index.core import Settings
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        Settings.node_parser = parser
        
        print(f"Indexing {len(documents)} documents with custom chunking...")
        self.index = VectorStoreIndex.from_documents(documents)
        self.index.storage_context.persist(persist_dir=str(Path(self.storage_dir) / "index_store"))
        print("✓ Index created and persisted")
    
    def chat(self, question: str) -> str:
        """Chat with conversation memory"""
        if self.index is None:
            raise ValueError("No index created. Call ingest_urls() first.")
        
        query_engine = self.index.as_chat_engine()
        response = query_engine.chat(question)
        
        self.conversation_history.append({
            "question": question,
            "response": str(response)
        })
        
        return response
    
    def batch_query(self, questions: List[str]) -> List[Dict]:
        """Process multiple questions and return answers with sources"""
        results = []
        for question in questions:
            result = self.query(question)
            results.append({
                "question": question,
                "answer": result['answer'],
                "sources": result['sources']
            })
        return results
    
    def get_conversation_history(self) -> List[Dict]:
        """Get chat conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
