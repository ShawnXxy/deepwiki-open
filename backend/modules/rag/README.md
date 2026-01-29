# RAG Module

The RAG (Retrieval-Augmented Generation) module provides the core functionality for embedding documents, retrieving relevant context, and generating AI-powered answers based on repository content.

## Module Structure

```
modules/rag/
├── __init__.py       # Module exports and aliases
├── retriever.py      # Main RAG component class
├── database.py       # Document database management
├── document.py       # Document processing and embedding
├── memory.py         # Conversation history management
├── answer.py         # RAG answer dataclass
└── utils.py          # Token counting and file utilities
```

## Components

### retriever.py - RAG Component

The main `RAG` class orchestrates document retrieval and answer generation:

```python
class RAG(adal.Component):
    def __init__(self, provider=None, model=None):
        # Always uses Azure OpenAI
        self.memory = Memory()
        self.embedder = get_embedder()
        self.db_manager = DatabaseManager()
        self.generator = adal.Generator(...)
    
    def prepare_retriever(self, repo_url, ...):
        # Loads/creates document database and FAISS index
        
    def retrieve(self, query, top_k=5):
        # Retrieves relevant documents using FAISS
        
    def call(self, query):
        # Full RAG pipeline: retrieve + generate
```

Key methods:
- `prepare_retriever()`: Downloads repo, processes documents, creates embeddings
- `_validate_and_filter_embeddings()`: Filters out documents with invalid embeddings
- FAISS retriever for semantic search

### database.py - DatabaseManager

Manages document storage with both local and blob storage support:

```python
class DatabaseManager:
    def prepare_database(self, repo_url, ...):
        # 1. Clone/pull repository
        # 2. Load existing vectors OR process fresh
        # 3. Return Document list
        
    def _create_repo(self, repo_url, ...):
        # Download repo and set up storage paths
        
    def _delete_existing_storage(self, ...):
        # Clean up for force_reprocess
```

Storage decision flow:
1. Check for existing JSON vectors in `vectors/{owner}_{repo}_{branch}/`
2. If not found, check for legacy `.pkl` database
3. If neither exists, process fresh from source files

### document.py - Document Processing

Handles reading, splitting, and embedding source files:

```python
def read_all_documents(path, repo_url, ...) -> List[Document]:
    # Recursively reads files with inclusion/exclusion filters
    # Returns Document objects with metadata
    
def transform_documents_and_save_as_json(documents, ...):
    # 1. Split documents into chunks
    # 2. Generate embeddings for each chunk
    # 3. Save as individual JSON files per chunk
```

File processing order:
1. Code files (`.py`, `.js`, `.ts`, `.java`, etc.)
2. Documentation files (`.md`, `.txt`, `.rst`)

Each document includes metadata:
- `file_path`: Relative path in repository
- `url`: Direct link to file on hosting platform
- `type`: File extension
- `is_implementation`: Whether it's a main source file

### memory.py - Conversation Memory

Manages multi-turn conversation history:

```python
class Memory(adal.core.component.DataComponent):
    def __init__(self):
        self.current_conversation = CustomConversation()
    
    def call(self) -> Dict:
        # Returns dialog turns as dictionary
        
    def add_dialog_turn(self, user_query, assistant_response):
        # Adds a Q&A pair to history
```

Uses `DialogTurn` dataclass with:
- `UserQuery`: The user's question
- `AssistantResponse`: The assistant's answer
- `id`: UUID for tracking

### answer.py - RAGAnswer

Structured output format for RAG responses:

```python
@dataclass
class RAGAnswer(adal.DataClass):
    rationale: str  # Chain of thought (internal)
    answer: str     # Formatted markdown response
```

### utils.py - Utilities

Token counting and file reading utilities:

```python
def count_tokens(text, include_special=True) -> int:
    # Uses tiktoken for accurate token counting
    
def safe_read_file(file_path) -> str:
    # Auto-detects encoding (UTF-8, UTF-16, Latin-1)
    
MAX_EMBEDDING_TOKENS = 7500  # Safe limit for Azure OpenAI
MAX_INPUT_TOKENS = 7500      # Safe threshold for context
```

## Workflow

### Document Indexing Flow

```
┌─────────────┐     ┌────────────────┐     ┌─────────────────┐
│  Repository │────►│  git clone     │────►│  read_all_      │
│    URL      │     │  (git_ops.py)  │     │  documents()    │
└─────────────┘     └────────────────┘     └────────┬────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │  FileFilter     │
                                           │  (include/      │
                                           │   exclude)      │
                                           └────────┬────────┘
                                                    │
                    ┌───────────────────────────────┘
                    │
                    ▼
           ┌─────────────────┐     ┌─────────────────┐
           │  TextSplitter   │────►│  ToEmbeddings   │
           │  (chunk text)   │     │  (Azure OpenAI) │
           └─────────────────┘     └────────┬────────┘
                                            │
                    ┌───────────────────────┘
                    │
                    ▼
           ┌─────────────────┐     ┌─────────────────┐
           │  VectorStorage  │────►│  FAISS Index    │
           │  (JSON files)   │     │  (in memory)    │
           └─────────────────┘     └─────────────────┘
```

### Query Flow

```
┌─────────────┐     ┌────────────────┐     ┌─────────────────┐
│  User Query │────►│  FAISS         │────►│  Top-K          │
│             │     │  Retriever     │     │  Documents      │
└─────────────┘     └────────────────┘     └────────┬────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │  Context        │
                                           │  Formatting     │
                                           └────────┬────────┘
                                                    │
                    ┌───────────────────────────────┘
                    │
                    ▼
           ┌─────────────────┐     ┌─────────────────┐
           │  Prompt +       │────►│  Azure OpenAI   │
           │  Context +      │     │  (Streaming)    │
           │  History        │     └────────┬────────┘
           └─────────────────┘              │
                                            ▼
                                   ┌─────────────────┐
                                   │  RAGAnswer      │
                                   │  (Markdown)     │
                                   └─────────────────┘
```

## Vector Storage Format

Documents are stored as individual JSON files:

```
vectors/{owner}_{repo}_{branch}/
├── src/backend/main_001.json    # Chunk 1 of main.py
├── src/backend/main_002.json    # Chunk 2 of main.py
├── src/utils/helper_001.json    # Chunk 1 of helper.py
└── README_001.json              # Chunk 1 of README.md
```

Each JSON file contains:
```json
{
    "file_path": "src/backend/main.py",
    "chunk_index": 0,
    "total_chunks": 5,
    "text": "chunk text content",
    "vector": [0.123, 0.456, ...],
    "meta_data": {
        "file_path": "src/backend/main.py",
        "type": "py",
        "url": "https://dev.azure.com/org/project/_git/repo?path=/src/backend/main.py"
    }
}
```

## Usage Example

```python
from backend.modules.rag import RAG

# Initialize RAG (always uses Azure OpenAI)
rag = RAG(provider="azure", model="gpt-4o")

# Prepare retriever for an Azure DevOps repository
rag.prepare_retriever(
    repo_url_or_path="https://dev.azure.com/org/project/_git/repo",
    type="azuredevops",
    token="your-pat-token",
    branch="main"
)

# Query the repository
response = rag.call("How does authentication work?")
print(response.answer)
```

## Dependencies

- `adalflow`: LLM and RAG framework
- `faiss-cpu`: Vector similarity search
- `tiktoken`: Token counting
- `backend.clients.azureai_client`: Azure OpenAI client
- `backend.clients.vector_storage`: JSON vector storage
- `backend.tools.embedder`: Embedding utilities
