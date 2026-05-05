# RAG Knowledge Base Q&A System

A production-ready Retrieval Augmented Generation (RAG) system for building internal knowledge base Q&A chatbots using LlamaIndex, Llama2, and web URLs.

## Features

- 🌐 **Web Scraping**: Automatically fetch and index web pages from URLs
- 💾 **Persistent Storage**: Cache indexed documents locally for fast reuse
- 📚 **Vector Search**: Fast semantic search using embeddings
- 🔗 **Source Citations**: Track which URLs answers came from
- 🦙 **Local LLM**: Run Llama2 locally via Ollama (no API keys needed)
- 🔄 **Automatic Chunking**: Intelligently split large documents
- 💬 **Chat Interface**: Interactive conversation with your knowledge base
- 🚀 **REST API**: Optional FastAPI endpoints for production deployment

## Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) installed
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/Jman1132/rag-knowledge-base.git
cd rag-knowledge-base

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Setup Ollama

```bash
# Install Ollama from https://ollama.ai
# Start Ollama in a terminal
ollama serve

# In another terminal, pull the model
ollama pull llama2
```

### Run

```bash
# Run advanced version with persistent storage (recommended)
python rag_advanced.py

# Or use the California Penal Code specialized version
python rag_california_penal_code.py
```

## Usage

### Basic Usage (Step 1)

```bash
python rag_basic.py
```

Simple implementation without persistent storage.

### Advanced Usage with Persistence (Step 3)

```bash
python rag_advanced.py
```

Features:
- Saves indexed documents to `./storage` directory
- Loads cached index on restart (much faster!)
- Shows source URLs for each answer
- Interactive Q&A loop

### REST API

```bash
python rag_with_api.py
```

Then query via HTTP:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the definition of murder?"}'
```

API Documentation: http://localhost:8000/docs

## Configuration

Edit `config.py` to customize:

```python
# LLM Settings
LLM_MODEL = "llama2"
LLM_BASE_URL = "http://localhost:11434"

# Embedding Settings
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# RAG Parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 20
SIMILARITY_TOP_K = 3
```

## Example Queries

For California Penal Code Knowledge Base:

- "What is the definition of murder in California?"
- "What are the degrees of murder?"
- "What is the penalty for first-degree murder?"
- "Explain voluntary vs involuntary manslaughter"
- "What constitutes implied malice?"
- "What is the difference between murder and manslaughter?"

## Knowledge Base

Currently configured to pull from:

- **Part 1**: California Penal Code - Of Crimes and Punishments (§25-680.4)
- **Title 8, Chapter 1**: Homicide (§187-199)

To customize:

1. Edit the URLs in `rag_advanced.py` or `config.py`
2. Run with `force_refresh=True` to re-index

## Troubleshooting

### Connection refused on localhost:11434

Make sure Ollama is running:

```bash
ollama serve
```

### Model not found

Pull the model:

```bash
ollama pull llama2
```

### Out of memory

Use a smaller model:

```bash
ollama pull mistral
```

Then update `config.py`: `LLM_MODEL = "mistral"`

### Slow performance

- Reduce `CHUNK_SIZE` in config.py
- Use `similarity_top_k=2` instead of 3
- Consider using GPU-enabled Ollama setup

## File Structure

```
rag-knowledge-base/
├── rag_system.py              # Core RAG classes
├── rag_basic.py               # Simple example
├── rag_advanced.py            # Advanced with storage
├── rag_california_penal_code.py  # Legal code specialized
├── rag_with_api.py            # REST API endpoints
├── config.py                  # Configuration
├── requirements.txt           # Dependencies
├── README.md                  # This file
└── storage/                   # Auto-created for index
    └── index_store/
```

## Performance Tips

1. **First Run**: Takes longer (fetches + indexes URLs)
2. **Subsequent Runs**: Instant (loads from cache)
3. **Force Refresh**: Use when URLs are updated
4. **Batch Operations**: Process multiple queries in one session
5. **GPU Acceleration**: Configure Ollama to use GPU for faster inference

## Production Deployment

1. Use REST API (`rag_with_api.py`)
2. Add authentication for API endpoints
3. Use a vector database (Pinecone, Weaviate) instead of local storage
4. Add comprehensive logging
5. Monitor performance metrics
6. Cache frequently asked questions

## Advanced Features

### Custom Chunking

```python
from rag_system import EnhancedRAG

rag = EnhancedRAG(llm=llm, embed_model=embed_model)
rag.ingest_urls_with_chunking(
    urls=urls,
    chunk_size=1024,
    chunk_overlap=50
)
```

### Chat with Memory

```python
response = rag.chat("Tell me about homicide")
response = rag.chat("What are the penalties?")  # Remembers context
```

### Batch Processing

```python
questions = [
    "What is murder?",
    "What is manslaughter?",
    "What are the penalties?"
]
results = rag.batch_query(questions)
```

## Resources

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Ollama](https://ollama.ai)
- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [HuggingFace Embeddings](https://huggingface.co/models)

## License

MIT License - see LICENSE file for details

## Support

For issues or questions, please open a GitHub issue.

---

**Happy RAG Building! 🚀**
