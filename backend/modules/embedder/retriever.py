"""
RAG (Retrieval Augmented Generation) component.

Provides the main RAG class for document retrieval and answer generation.
Supports two retrieval backends:
  - FAISS (local/Docker): loads vectors into memory, nearest-neighbor search
  - AI Search (cloud): hybrid text+vector query against Azure AI Search index
"""

import logging
from typing import List, Tuple

import adalflow as adal
from adalflow.components.retriever.faiss_retriever import FAISSRetriever

from backend.config import configs, is_search_configured
from backend.clients.embedding_client import get_embedder
from backend.modules.embedder.memory import Memory
from backend.modules.embedder.response import RAGAnswer
from backend.modules.embedder.indexer import DatabaseManager
from backend.promptstore import RAG_SYSTEM_PROMPT as system_prompt, RAG_TEMPLATE

logger = logging.getLogger(__name__)

# Maximum token limit for embedding models
MAX_INPUT_TOKENS = 7500  # Safe threshold below 8192 token limit


class RAG(adal.Component):
    """RAG with one repo.
    If you want to load a new repos, call prepare_retriever(repo_url_or_path) first."""

    def __init__(self, provider=None, model=None, use_s3: bool = False):  # noqa: F841 - use_s3 is kept for compatibility
        """
        Initialize the RAG component.

        Args:
            provider: Model provider (ignored, always uses Azure OpenAI)
            model: Model name to use with Azure OpenAI
            use_s3: Whether to use S3 for database storage (default: False)
        """
        super().__init__()

        # Always use Azure OpenAI as the provider
        from backend.config import is_azure_openai_configured

        self.provider = "azure"
        self.model = model
        
        if not is_azure_openai_configured():
            logger.warning("Azure OpenAI is not configured. Please set the required environment variables.")

        # Always use Azure embedder type
        self.embedder_type = 'azure'
        self.is_ollama_embedder = False  # Backward compatibility

        # Initialize components
        self.memory = Memory()
        self.embedder = get_embedder()

        self.initialize_db_manager()

        # Set up the output parser
        data_parser = adal.DataClassParser(data_class=RAGAnswer, return_data_class=True)

        # Format instructions to ensure proper output structure
        format_instructions = data_parser.get_output_format_str() + """

IMPORTANT FORMATTING RULES:
1. DO NOT include your thinking or reasoning process in the output
2. Provide only the final, polished answer
3. DO NOT include ```markdown fences at the beginning or end of your answer
4. DO NOT wrap your response in any kind of fences
5. Start your response directly with the content
6. The content will already be rendered as markdown
7. Do not use backslashes before special characters like [ ] { } in your answer
8. When listing tags or similar items, write them as plain text without escape characters
9. For pipe characters (|) in text, write them directly without escaping them"""

        # Get model configuration based on provider and model
        from backend.config import get_model_config, get_azure_ai_client
        generator_config = get_model_config(self.provider, self.model)

        # Use shared Azure AI client instance (singleton)
        model_client = get_azure_ai_client(self.model)

        # Set up the main generator
        self.generator = adal.Generator(
            template=RAG_TEMPLATE,
            prompt_kwargs={
                "output_format_str": format_instructions,
                "conversation_history": self.memory(),
                "system_prompt": system_prompt,
                "contexts": None,
            },
            model_client=model_client,
            model_kwargs=generator_config["model_kwargs"],
            output_processors=data_parser,
        )

    def initialize_db_manager(self):
        """Initialize the database manager with local storage"""
        self.db_manager = DatabaseManager()
        self.transformed_docs = []
        self.use_cloud_search = False
        self.cloud_index_name = None

    def _validate_and_filter_embeddings(self, documents: List) -> List:
        """
        Validate embeddings and filter out documents with invalid or mismatched embedding sizes.

        Single-pass: collects size counts and valid docs simultaneously,
        then filters by the most common size.

        Args:
            documents: List of documents with embeddings

        Returns:
            List of documents with valid embeddings of consistent size
        """
        if not documents:
            logger.warning("No documents provided for embedding validation")
            return []

        embedding_sizes = {}  # size -> count
        docs_by_size = {}     # size -> list of docs

        for i, doc in enumerate(documents):
            if not hasattr(doc, 'vector') or doc.vector is None:
                logger.warning(f"Document {i} has no embedding vector, skipping")
                continue

            try:
                if isinstance(doc.vector, list):
                    embedding_size = len(doc.vector)
                elif hasattr(doc.vector, 'shape'):
                    embedding_size = (
                        doc.vector.shape[0]
                        if len(doc.vector.shape) == 1
                        else doc.vector.shape[-1]
                    )
                elif hasattr(doc.vector, '__len__'):
                    embedding_size = len(doc.vector)
                else:
                    logger.warning(
                        f"Document {i} has invalid embedding vector "
                        f"type: {type(doc.vector)}, skipping"
                    )
                    continue

                if embedding_size == 0:
                    logger.warning(f"Document {i} has empty embedding vector, skipping")
                    continue

                embedding_sizes[embedding_size] = (
                    embedding_sizes.get(embedding_size, 0) + 1
                )
                if embedding_size not in docs_by_size:
                    docs_by_size[embedding_size] = []
                docs_by_size[embedding_size].append(doc)

            except Exception as e:
                logger.warning(
                    f"Error checking embedding size for document {i}: "
                    f"{str(e)}, skipping"
                )
                continue

        if not embedding_sizes:
            logger.error("No valid embeddings found in any documents")
            return []

        # Pick the most common size
        target_size = max(
            embedding_sizes.keys(), key=lambda k: embedding_sizes[k]
        )
        logger.info(
            f"Target embedding size: {target_size} "
            f"(found in {embedding_sizes[target_size]} documents)"
        )

        for size, count in embedding_sizes.items():
            if size != target_size:
                logger.warning(
                    f"Found {count} documents with incorrect "
                    f"embedding size {size}, will be filtered out"
                )

        valid_documents = docs_by_size.get(target_size, [])

        logger.info(
            f"Embedding validation complete: "
            f"{len(valid_documents)}/{len(documents)} documents "
            f"have valid embeddings"
        )

        if len(valid_documents) == 0:
            logger.error(
                "No documents with valid embeddings remain after filtering"
            )
        elif len(valid_documents) < len(documents):
            filtered_count = len(documents) - len(valid_documents)
            logger.warning(
                f"Filtered out {filtered_count} documents "
                f"due to embedding issues"
            )

        return valid_documents

    def prepare_retriever(
        self,
        repo_url_or_path: str,
        type: str = "github",
        access_token: str = None,
        branch: str = None,
        excluded_dirs: List[str] = None,
        excluded_files: List[str] = None,
        included_dirs: List[str] = None,
        included_files: List[str] = None,
        force_reprocess: bool = False
    ):
        """
        Prepare the retriever for a repository.

        In cloud mode (AI Search configured) with force_reprocess=False:
            Skips local vector loading and FAISS — queries go to AI Search.
        In cloud mode with force_reprocess=True (processor run):
            Must embed documents first (FAISS path), then AI Search is used
            after vectors are pushed to the search index.
        In local/Docker mode: loads vectors and builds FAISS index as before.
        """
        self.initialize_db_manager()
        self.repo_url_or_path = repo_url_or_path

        # --- Cloud mode: use AI Search for READ-ONLY retrieval ---
        # Skip this shortcut when force_reprocess=True (processor must embed first)
        if is_search_configured() and not force_reprocess:
            from backend.clients.search_client import (
                get_index_name, index_exists,
            )
            # Derive index name from repo URL
            from backend.processor.code_processor import _extract_owner_repo
            owner, repo = _extract_owner_repo(repo_url_or_path)
            branch_suffix = branch.strip() if branch and branch.strip() else 'main'
            idx_name = get_index_name(owner, repo, branch_suffix)

            if index_exists(idx_name):
                self.use_cloud_search = True
                self.cloud_index_name = idx_name
                logger.info(
                    f"[RAG] Cloud mode: using AI Search index '{idx_name}'"
                )
                return
            else:
                logger.warning(
                    f"[RAG] AI Search configured but index '{idx_name}' "
                    f"not found — falling back to FAISS"
                )

        # --- Local/Docker mode: load vectors + build FAISS ---
        self.transformed_docs = self.db_manager.prepare_database(
            repo_url_or_path,
            type,
            access_token,
            branch=branch,
            embedder_type=self.embedder_type,
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_dirs=included_dirs,
            included_files=included_files,
            force_reprocess=force_reprocess
        )
        logger.info(f"Loaded {len(self.transformed_docs)} documents for retrieval")

        # Validate and filter embeddings to ensure consistent sizes
        self.transformed_docs = self._validate_and_filter_embeddings(self.transformed_docs)

        if not self.transformed_docs:
            # Check if we had documents initially but they were filtered out due to empty embeddings
            raise ValueError("No valid documents with embeddings found. This usually means the embedding process failed. Check the logs for 'CRITICAL: Azure Embedding Failed' or similar errors. Verify your API key and endpoint configuration.")

        logger.info(f"Using {len(self.transformed_docs)} documents with valid embeddings for retrieval")

        try:
            # Filter out internal keys not accepted by FAISSRetriever
            faiss_kwargs = {
                k: v for k, v in configs["retriever"].items()
                if k in ("top_k",)
            }
            self.retriever = FAISSRetriever(
                **faiss_kwargs,
                embedder=self.embedder,
                documents=self.transformed_docs,
                document_map_func=lambda doc: doc.vector,
            )
            logger.info("FAISS retriever created successfully")

            # Strip embedding vectors from document list — FAISS has
            # its own copy.  Keeps text + meta_data for lookup but
            # frees ~12 KB per chunk (3072 floats × 4 bytes).
            for doc in self.transformed_docs:
                doc.vector = None
        except Exception as e:
            logger.error(f"Error creating FAISS retriever: {str(e)}")
            # Try to provide more specific error information
            if "All embeddings should be of the same size" in str(e):
                logger.error("Embedding size validation failed. This suggests there are still inconsistent embedding sizes.")
                # Log embedding sizes for debugging
                sizes = []
                for i, doc in enumerate(self.transformed_docs[:10]):  # Check first 10 docs
                    if hasattr(doc, 'vector') and doc.vector is not None:
                        try:
                            if isinstance(doc.vector, list):
                                size = len(doc.vector)
                            elif hasattr(doc.vector, 'shape'):
                                size = doc.vector.shape[0] if len(doc.vector.shape) == 1 else doc.vector.shape[-1]
                            elif hasattr(doc.vector, '__len__'):
                                size = len(doc.vector)
                            else:
                                size = "unknown"
                            sizes.append(f"doc_{i}: {size}")
                        except:
                            sizes.append(f"doc_{i}: error")
                logger.error(f"Sample embedding sizes: {', '.join(sizes)}")
            raise

    def call(self, query: str, language: str = "en") -> Tuple[List]:
        """
        Process a query using RAG.

        Uses AI Search in cloud mode, FAISS in local mode.

        Returns:
            Tuple of (RAGAnswer, retrieved_documents)
        """
        # --- Cloud path: AI Search hybrid query ---
        if self.use_cloud_search and self.cloud_index_name:
            return self._call_cloud(query, language)

        # --- Local path: FAISS ---
        try:
            retrieved_documents = self.retriever(query)

            # Fill in the documents
            retrieved_documents[0].documents = [
                self.transformed_docs[doc_index]
                for doc_index in retrieved_documents[0].doc_indices
            ]

            return retrieved_documents

        except Exception as e:
            logger.error(f"Error in RAG call: {str(e)}")

            # Create error response
            error_response = RAGAnswer(
                rationale="Error occurred while processing the query.",
                answer="I apologize, but I encountered an error while "
                "processing your question. Please try again or "
                "rephrase your question."
            )
            return error_response, []

    def _extract_query_vector(self, query: str):
        """Embed a query and extract the vector for AI Search."""
        embedding_output = self.embedder([query])
        # EmbedderOutput has .data list of Embedding objects
        if hasattr(embedding_output, 'data') and embedding_output.data:
            first = embedding_output.data[0]
            if hasattr(first, 'embedding'):
                return first.embedding
            elif isinstance(first, list):
                return first
        # Fallback: try direct list access
        if isinstance(embedding_output, list) and embedding_output:
            if hasattr(embedding_output[0], 'embedding'):
                return embedding_output[0].embedding
            return embedding_output[0]
        logger.warning("[RAG] Could not extract embedding vector")
        return None

    def _call_cloud(self, query: str, language: str = "en"):
        """Execute retrieval via AI Search (cloud mode)."""
        from backend.clients.search_client import search_as_documents

        try:
            query_vector = self._extract_query_vector(query)
            top_k = configs.get("retriever", {}).get("top_k", 40)

            docs = search_as_documents(
                index_name=self.cloud_index_name,
                query=query,
                top_k=top_k,
                vector=query_vector,
            )
            logger.info(
                f"[RAG] Cloud search returned {len(docs)} results"
            )

            # Wrap in a result structure compatible with FAISS output
            class _CloudResult:
                def __init__(self, documents):
                    self.documents = documents
                    self.doc_indices = list(range(len(documents)))

            return [_CloudResult(docs)]

        except Exception as e:
            logger.error(f"Error in cloud RAG call: {e}")

            class _EmptyResult:
                def __init__(self):
                    self.documents = []
                    self.doc_indices = []

            return [_EmptyResult()]

    def call_with_file_filter(
        self,
        query: str,
        file_paths: List[str],
        top_k: int = None,
        language: str = "en",
    ) -> Tuple[List]:
        """
        Retrieve chunks with priority for specific files.

        Strategy (both local and cloud):
        1. Collect chunks from declared relevant files
        2. Run semantic search for supplementary context
        3. Merge: file-filtered chunks first, then semantic (deduplicated)
        """
        # --- Cloud path ---
        if self.use_cloud_search and self.cloud_index_name:
            return self._call_with_file_filter_cloud(
                query, file_paths, top_k, language
            )

        # --- Local path: FAISS ---
        try:
            # Step 1: Get ALL chunks from declared relevant files
            file_chunks = [
                doc for doc in self.transformed_docs
                if doc.meta_data.get('file_path', '') in file_paths
            ]
            logger.info(
                f"[RAG] File-filtered retrieval: {len(file_chunks)} "
                f"chunks from {len(file_paths)} declared files"
            )

            # Step 2: Semantic search for supplementary context
            original_top_k = self.retriever.top_k
            if top_k:
                self.retriever.top_k = top_k
            try:
                semantic_results = self.retriever(query)
            finally:
                self.retriever.top_k = original_top_k

            semantic_docs = [
                self.transformed_docs[idx]
                for idx in semantic_results[0].doc_indices
            ]

            # Step 3: Merge — file chunks first, then semantic (deduplicated)
            seen_ids = {id(doc) for doc in file_chunks}
            for doc in semantic_docs:
                if id(doc) not in seen_ids:
                    file_chunks.append(doc)
                    seen_ids.add(id(doc))

            logger.info(
                f"[RAG] Merged result: {len(file_chunks)} total chunks"
            )
            semantic_results[0].documents = file_chunks
            return semantic_results

        except Exception as e:
            logger.error(f"Error in file-filtered RAG call: {str(e)}")
            # Fallback to regular retrieval
            return self.call(query, language)

    def _call_with_file_filter_cloud(
        self,
        query: str,
        file_paths: List[str],
        top_k: int = None,
        language: str = "en",
    ):
        """File-priority retrieval via AI Search (cloud mode).

        1. Query AI Search filtered to declared files
        2. Query AI Search unfiltered for semantic context
        3. Merge: file-filtered first, then semantic (deduplicated)
        """
        from backend.clients.search_client import search_as_documents

        try:
            effective_top_k = (
                top_k or configs.get("retriever", {}).get("top_k", 40)
            )

            # Embed the query once
            query_vector = self._extract_query_vector(query)

            # Step 1: Get chunks from declared files via filter
            file_chunks = []
            if file_paths:
                # Build OData filter: filepath eq 'a' or filepath eq 'b'
                conditions = [
                    f"filepath eq '{fp}'" for fp in file_paths
                ]
                filter_expr = " or ".join(conditions)
                file_chunks = search_as_documents(
                    index_name=self.cloud_index_name,
                    query=query,
                    top_k=effective_top_k,
                    vector=query_vector,
                    filter_expr=filter_expr,
                )
            logger.info(
                f"[RAG] Cloud file-filtered: {len(file_chunks)} chunks "
                f"from {len(file_paths)} declared files"
            )

            # Step 2: Unfiltered semantic search
            semantic_docs = search_as_documents(
                index_name=self.cloud_index_name,
                query=query,
                top_k=effective_top_k,
                vector=query_vector,
            )

            # Step 3: Merge — file chunks first, deduplicate
            seen_texts = {doc.text for doc in file_chunks}
            for doc in semantic_docs:
                if doc.text not in seen_texts:
                    file_chunks.append(doc)
                    seen_texts.add(doc.text)

            logger.info(
                f"[RAG] Cloud merged: {len(file_chunks)} total chunks"
            )

            class _CloudResult:
                def __init__(self, documents):
                    self.documents = documents
                    self.doc_indices = list(range(len(documents)))

            return [_CloudResult(file_chunks)]

        except Exception as e:
            logger.error(f"Error in cloud file-filtered RAG: {e}")
            return self._call_cloud(query, language)
