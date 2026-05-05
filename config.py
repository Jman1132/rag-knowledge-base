#!/usr/bin/env python3
"""
Configuration for RAG System

Edit this file to customize RAG parameters.
"""

# LLM Settings
LLM_MODEL = "llama2"
LLM_BASE_URL = "http://localhost:11434"

# Embedding Settings
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# RAG Parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 20
SIMILARITY_TOP_K = 3
STORAGE_DIR = "./storage"

# California Penal Code URLs
CALIFORNIA_PENAL_CODE_URLS = [
    "https://law.justia.com/codes/california/code-pen/part-1/",
    "https://law.justia.com/codes/california/code-pen/part-1/title-8/chapter-1/"
]

# API Settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_DEBUG = False
