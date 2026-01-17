import hashlib
import logging
import time
from typing import List, Dict, Set, Optional, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import chromadb
import ollama
import chromadb
import ollama
from sentence_transformers import CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from tqdm import tqdm

from config import ProjectConfig, logger
from utils.gpu_lock import GPULock
from utils import memory
from core.services.rag_cache import RAGQueryCache

# Smart Router integration (lazy import)
_smart_router = None

def _get_smart_router():
    """Lazy load SmartRouter."""
    global _smart_router
    if _smart_router is None and ProjectConfig.SMART_ROUTER_ENABLED:
        try:
            from core.services.llm_router import get_router
            _smart_router = get_router()
        except ImportError:
            pass
    return _smart_router

class ObsidianRAG:
    """
    RAG Engine 2.1: Optimized for Incremental Indexing and local LLMs.
    Now supports semantic search for auto-linking.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path if db_path else ProjectConfig.CHROMA_DB_DIR
        self.embedding_model_name = ProjectConfig.OLLAMA_EMBEDDING_MODEL
        self.generation_model_name = ProjectConfig.OLLAMA_GENERATION_MODEL
        
        # Initialize Ollama client check (ensure model exists)
        try:
             # Basic check to see if model is available/pullable would happen here externally
             pass
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            raise e

        # Initialize Vector DB
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_or_create_collection(
            name="obsidian_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.logger = logging.getLogger("RAG-Engine")
        
        # Advanced Chunking: Header Splitter -> Recursive Splitter
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)
        
        # Text splitter optimized for Markdown and Code
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=ProjectConfig.RAG_CHUNK_SIZE,
            chunk_overlap=ProjectConfig.RAG_CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n#### ", "\n", " ", ""],
            keep_separator=True
        )
        
        # Reranking Model (Lightweight Cross-Encoder)
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            self.logger.warning(f"Failed to load CrossEncoder: {e}")
            self.cross_encoder = None

        # Query Result Cache (to avoid expensive HyDE generation)
        cache_dir = ProjectConfig.BASE_DIR / "data" / "rag_cache"
        self.query_cache = RAGQueryCache(
            cache_dir=cache_dir,
            max_size=100,  # Cache 100 recent queries
            ttl_hours=24  # 24-hour TTL
        )

    def _get_file_hash(self, file_path: Path) -> str:
        """Generates MD5 hash for content validation."""
        return hashlib.md5(file_path.read_bytes()).hexdigest()

    def _get_indexed_metadata(self) -> Dict[str, Dict[str, str]]:
        """Builds a map of indexed files for incremental update logic."""
        results = self.collection.get(include=['metadatas'])
        indexed = {}
        if results and results['metadatas']:
            for meta in results['metadatas']:
                fname = meta.get('filename')
                if fname:
                    indexed[fname] = {
                        "hash": meta.get("file_hash"),
                        "mtime": str(meta.get("mtime", ""))
                    }
        return indexed

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings using Ollama.
        Auto-truncates text to avoid context length limits (512 tokens).
        """
        truncation_limit = 1500  # Safe limit for ~512 token context
        truncated_texts = [t[:truncation_limit] for t in texts]
        
        try:
            # Ollama python library 'embed' or 'embeddings'
            response = ollama.embed(model=self.embedding_model_name, input=truncated_texts)
            return response['embeddings']
        except Exception as e:
            self.logger.error(f"Embedding failed: {e}", extra={"tags": "RAG-ERROR"})
            # Fallback: return zero vectors or raise? Raising is safer for consistency.
            raise e

    def _process_single_file(self, file_path: Path, file_hash_cache: Dict[str, str]) -> Tuple[int, str]:
        """
        Process a single file for indexing.

        Returns:
            (num_chunks, filename) tuple
        """
        file_name = file_path.name

        try:
            mtime = str(file_path.stat().st_mtime)
            file_hash = self._get_file_hash(file_path)
            file_hash_cache[file_name] = file_hash

            # Check if file needs update
            indexed_map = file_hash_cache.get('_indexed_map', {})
            if file_name in indexed_map:
                if indexed_map[file_name]["hash"] == file_hash:
                    return (0, file_name)  # Skip unchanged file

            # Delete old version if exists
            if file_name in indexed_map:
                self.collection.delete(where={"filename": file_name})

            # Read and chunk file
            content = file_path.read_text(encoding='utf-8')

            # Split by headers
            header_splits = self.header_splitter.split_text(content)

            # Further split by recursive character splitter
            chunks = []
            chunk_metadatas = []

            for split in header_splits:
                sub_chunks = self.splitter.split_text(split.page_content)
                for i, sub_chunk in enumerate(sub_chunks):
                    chunks.append(sub_chunk)
                    meta = split.metadata.copy()
                    meta.update({
                        "filename": file_name,
                        "file_hash": file_hash,
                        "mtime": mtime,
                        "chunk_index": i,
                        "source": str(file_path)
                    })
                    chunk_metadatas.append(meta)

            if not chunks:
                return (0, file_name)

            # Generate embeddings and upsert
            ids = [f"{file_name}_{i}_{file_hash[:6]}" for i in range(len(chunks))]
            embeddings = self._get_embeddings(chunks)

            self.collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=chunk_metadatas)

            return (len(chunks), file_name)

        except Exception as e:
            self.logger.error(f"Failed to process {file_name}: {e}")
            return (0, file_name)

    def index_vault(self, vault_path: Path, parallel: bool = True, max_workers: int = 4) -> int:
        """
        Performs Incremental Indexing of the Obsidian Vault.

        Args:
            vault_path: Path to Obsidian vault
            parallel: Use parallel processing (default: True)
            max_workers: Number of parallel workers (default: 4)

        Returns:
            Number of new chunks indexed
        """
        if not vault_path.exists():
            self.logger.error(f"Vault path not found: {vault_path}")
            return 0

        self.logger.info(f"Starting Incremental Indexing: {vault_path}", extra={"tags": "RAG-INDEX"})
        all_files = [f for f in vault_path.rglob("*.md") if not f.name.startswith('.')]
        indexed_map = self._get_indexed_metadata()

        # Shared cache for file hashes (used by worker threads)
        file_hash_cache = {'_indexed_map': indexed_map}

        new_chunks = 0
        current_filenames = set()

        if parallel and len(all_files) > 1:
            # Parallel processing
            workers = min(max_workers, os.cpu_count() or 4)
            self.logger.info(f"Using {workers} parallel workers for indexing")

            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit all files for processing
                future_to_file = {
                    executor.submit(self._process_single_file, f, file_hash_cache): f
                    for f in all_files
                }

                # Collect results with progress bar
                for future in tqdm(as_completed(future_to_file), total=len(all_files), desc="Indexing Vault"):
                    file_path = future_to_file[future]
                    try:
                        num_chunks, file_name = future.result()
                        new_chunks += num_chunks
                        current_filenames.add(file_name)
                    except Exception as e:
                        self.logger.error(f"Failed to process {file_path.name}: {e}")

        else:
            # Sequential processing (fallback or small vaults)
            for file_path in tqdm(all_files, desc="Indexing Vault"):
                try:
                    num_chunks, file_name = self._process_single_file(file_path, file_hash_cache)
                    new_chunks += num_chunks
                    current_filenames.add(file_name)
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path.name}: {e}")

        stale_files = set(indexed_map.keys()) - current_filenames
        if stale_files:
            for sf in stale_files:
                self.collection.delete(where={"filename": sf})

        return new_chunks

    def _generate_hypothetical_answer(self, query: str) -> str:
        """
        HyDE (Hypothetical Document Embeddings):
        Generates a fake technical document snippet to improve retrieval.

        Now uses SmartRouter for intelligent provider selection with fallback.
        """
        prompt = f"Jesteś ekspertem. Napisz hipotetyczny fragment dokumentacji technicznej, który odpowiada na pytanie: {query}. Nie odpowiadaj wprost, stwórz tekst, który mógłby się znaleźć w notatkach."

        # Try SmartRouter first
        router = _get_smart_router()
        if router is not None:
            try:
                from core.services.llm_router import TaskType
                result = router.execute(
                    task_type=TaskType.RAG_HYDE,
                    prompt=prompt,
                    timeout=30.0
                )
                if result.success and result.content:
                    return result.content
                self.logger.warning(f"SmartRouter HyDE failed: {result.error}")
            except Exception as e:
                self.logger.warning(f"SmartRouter HyDE error: {e}")

        # Fallback to direct Ollama call
        try:
            response = ollama.chat(
                model=ProjectConfig.OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                stream=False
            )
            return response['message']['content']
        except Exception as e:
            self.logger.warning(f"Ollama HyDE failed: {e}. Using fallback.")
            return f"[FALLBACK] {query}"

    def find_related_notes(self, text: str, n_results: int = 3, threshold: float = 0.4) -> List[Dict]:
        """
        [NEW] Semantic Search for Auto-Linking.
        Returns list of {filename, score} for concepts semantically similar to input text.
        """
        try:
            query_embed = self._get_embeddings([text])[0]
            results = self.collection.query(
                query_embeddings=[query_embed], 
                n_results=n_results
            )
            
            related = []
            if results['documents'] and results['distances']:
                for i, doc in enumerate(results['documents'][0]):
                    dist = results['distances'][0][i]
                    meta = results['metadatas'][0][i]
                    if dist < threshold: 
                        related.append({
                            "filename": meta['filename'],
                            "score": dist,
                            "excerpt": doc[:100] + "..."
                        })
            return related
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []

    def query(self, question: str, history: List[Dict] = None, n_results=5, model_name=None, stream=False):
        """Retrieves context and generates response with Reranking and HyDE."""
        model = model_name or getattr(self, 'generation_model_name', ProjectConfig.OLLAMA_GENERATION_MODEL)
        lock = GPULock()
        try:
            # Check cache first (avoids expensive HyDE generation)
            cached_result = self.query_cache.get(question, n_results)
            if cached_result:
                response_text, sources = cached_result
                self.logger.info(f"[CACHE HIT] Query: {question[:50]}... (saved ~30s)")

                # Format as Ollama response for consistency
                response = {"message": {"content": response_text}}

                if stream:
                    def cached_gen():
                        yield response
                    return cached_gen(), set(sources)

                return [response], set(sources)

            self.logger.info(f"[CACHE MISS] Query: {question[:50]}...")

            # 1. Retrieval (High Recall) with HyDE
            initial_k = 20
            
            # Generate hypothetical answer for better retrieval alignment
            with lock:
                 hypothetical_doc = self._generate_hypothetical_answer(question)
            
            self.logger.info(f"HyDE Document generated: {hypothetical_doc[:50]}...")
            
            # Embed the hypothetical document, not the question
            q_embed = self._get_embeddings([hypothetical_doc])[0]
            
            results = self.collection.query(query_embeddings=[q_embed], n_results=initial_k)
            
            if not results['documents'] or not results['documents'][0]:
                context = "Brak odpowiednich notatek w bazie wiedzy."
                sources = set()
            else:
                docs = results['documents'][0]
                metas = results['metadatas'][0]
                
                # 2. Reranking (High Precision)
                if self.cross_encoder:
                    pairs = [[question, doc] for doc in docs]
                    scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
                    
                    # Sort by score descending
                    ranked = sorted(zip(docs, metas, scores), key=lambda x: x[2], reverse=True)
                    
                    # Take top n_results
                    top_k = ranked[:n_results]
                    
                    # Format context with scores
                    context = "\n".join([f"--- DOKUMENT: {m['filename']} (Relevance: {s:.2f}) ---\n{d}" for d, m, s in top_k])
                    sources = {m['filename'] for d, m, s in top_k}
                else:
                    # Fallback (Just take top n_results from Vector DB)
                    # Vector DB results are already sorted by distance (but check if chroma returns sorted?)
                    # Chroma usually returns sorted by distance.
                    top_k_docs = docs[:n_results]
                    top_k_metas = metas[:n_results]
                    context = "\n".join([f"--- DOKUMENT: {m['filename']} ---\n{d}" for d, m in zip(top_k_docs, top_k_metas)])
                    sources = {m['filename'] for m in top_k_metas}

            system_msg = f"KONTEKST:\n{context}\n\nOdpowiedz na pytanie na podstawie kontekstu."

            messages = [{'role': 'system', 'content': system_msg}]
            if history:
                messages.extend([m for m in history if m.get('role') in ['user', 'assistant']])
            messages.append({'role': 'user', 'content': question})

            # Try SmartRouter first for non-streaming
            router = _get_smart_router()
            if router is not None and not stream:
                try:
                    from core.services.llm_router import TaskType, Complexity

                    # Estimate complexity based on context length
                    complexity = Complexity.SIMPLE if len(context) < 2000 else (
                        Complexity.MEDIUM if len(context) < 5000 else Complexity.COMPLEX
                    )

                    result = router.execute(
                        task_type=TaskType.RAG_CHAT,
                        prompt=question,
                        system_prompt=system_msg,
                        complexity=complexity,
                        context_length=len(context)
                    )

                    if result.success:
                        response_text = result.content
                        sources_list = list(sources)
                        self.query_cache.put(question, n_results, response_text, sources_list)
                        self.logger.info(f"[CACHE STORED] Query cached (via SmartRouter: {result.provider_used.value})")

                        response = {"message": {"content": response_text}}
                        return [response], sources

                    self.logger.warning(f"SmartRouter RAG failed: {result.error}, falling back to Ollama")
                except Exception as e:
                    self.logger.warning(f"SmartRouter error: {e}, falling back to Ollama")

            # Fallback: Call Ollama directly
            memory.release_vram()
            with lock:
                response = ollama.chat(model=model, messages=messages, stream=stream)

            if stream:
                # Don't cache streaming responses (too complex)
                return response, sources

            # Store in cache (non-streaming only)
            response_text = response['message']['content']
            sources_list = list(sources)
            self.query_cache.put(question, n_results, response_text, sources_list)
            self.logger.info(f"[CACHE STORED] Query cached for future use")

            # Non-streaming: return as list for consistency
            return [response], sources

        except Exception as e:
            self.logger.error(f"LLM Query failed: {e}")
            error_msg = {"message": {"content": f"⚠️ Błąd systemowy: {e}"}}
            if stream:
                def error_gen():
                    yield error_msg
                return error_gen(), set()
            return [error_msg], set()

            