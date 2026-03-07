"""Vector store management for the RAG pipeline.

Manages a ChromaDB (or FAISS) vector store for document retrieval.
Handles adding documents, persisting the store, and querying.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages a ChromaDB vector store for RAG document retrieval.

    Provides methods to add document chunks with embeddings, persist
    the store to disk, and perform similarity search queries.

    Attributes:
        store_type: Backend type (``"chroma"`` or ``"faiss"``).
        store_path: Filesystem path for persistence.
        collection_name: ChromaDB collection name.
        embedding_dim: Dimension of stored embeddings.
    """

    def __init__(
        self,
        store_type: str = "chroma",
        store_path: str = "./chroma_db",
        collection_name: str = "energy_trading_docs",
        embedding_dim: int = 384,
    ) -> None:
        """Initialise the vector store manager.

        Args:
            store_type: Storage backend (``"chroma"`` or ``"faiss"``).
            store_path: Directory path for persisting the vector store.
            collection_name: Name of the ChromaDB collection.
            embedding_dim: Dimensionality of document embeddings.
        """
        self.store_type = store_type
        self.store_path = Path(store_path)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self._client = None
        self._collection = None
        self._faiss_index = None
        self._faiss_docs: list[dict] = []
        logger.info("VectorStoreManager initialised (type=%s, path=%s)", store_type, store_path)

    def _init_chroma(self) -> None:
        """Initialise ChromaDB client and collection."""
        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(self.store_path))
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB collection '%s' ready (%d documents)",
                self.collection_name,
                self._collection.count(),
            )
        except ImportError:
            logger.error("chromadb not installed")
            raise

    def _init_faiss(self) -> None:
        """Initialise an in-memory FAISS index."""
        try:
            import faiss

            self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            logger.info("FAISS IndexFlatIP initialised (dim=%d)", self.embedding_dim)
        except ImportError:
            logger.error("faiss-cpu not installed")
            raise

    def _ensure_initialised(self) -> None:
        """Ensure the backend is initialised."""
        if self.store_type == "chroma" and self._collection is None:
            self._init_chroma()
        elif self.store_type == "faiss" and self._faiss_index is None:
            self._init_faiss()

    def add_chunks(self, chunks: list[dict]) -> None:
        """Add embedded document chunks to the vector store.

        Args:
            chunks: List of chunk dicts with ``text``, ``embedding``,
                and metadata keys.

        Raises:
            ValueError: If chunks do not contain embeddings.
        """
        if not chunks:
            logger.warning("No chunks provided to add_chunks")
            return
        if "embedding" not in chunks[0]:
            raise ValueError("Chunks must have 'embedding' key. Run EmbeddingGenerator first.")

        self._ensure_initialised()

        if self.store_type == "chroma":
            self._add_to_chroma(chunks)
        elif self.store_type == "faiss":
            self._add_to_faiss(chunks)

    def _add_to_chroma(self, chunks: list[dict]) -> None:
        """Add chunks to ChromaDB."""
        ids = [
            f"{c.get('filename', 'doc')}_p{c.get('page_number', 0)}_c{c.get('chunk_index', i)}"
            for i, c in enumerate(chunks)
        ]
        texts = [c["text"] for c in chunks]
        embeddings = [
            c["embedding"].tolist() if isinstance(c["embedding"], np.ndarray) else c["embedding"]
            for c in chunks
        ]
        metadatas = [
            {
                "source": c.get("source", ""),
                "filename": c.get("filename", ""),
                "page_number": str(c.get("page_number", 0)),
                "chunk_index": str(c.get("chunk_index", 0)),
            }
            for c in chunks
        ]
        self._collection.add(  # type: ignore[union-attr]
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Added %d chunks to ChromaDB collection", len(chunks))

    def _add_to_faiss(self, chunks: list[dict]) -> None:
        """Add chunks to FAISS index."""
        import faiss

        embeddings = np.array([c["embedding"] for c in chunks], dtype=np.float32)
        faiss.normalize_L2(embeddings)
        self._faiss_index.add(embeddings)  # type: ignore[union-attr]
        self._faiss_docs.extend(chunks)
        logger.info("Added %d chunks to FAISS index", len(chunks))

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        """Retrieve the most similar document chunks for a query embedding.

        Args:
            query_embedding: Query vector of shape ``(embedding_dim,)``.
            top_k: Number of results to return.

        Returns:
            List of result dictionaries with ``text``, ``score``, and
            metadata keys, sorted by relevance score descending.
        """
        self._ensure_initialised()

        if self.store_type == "chroma":
            return self._query_chroma(query_embedding, top_k)
        if self.store_type == "faiss":
            return self._query_faiss(query_embedding, top_k)
        return []

    def _query_chroma(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        """Query ChromaDB for similar chunks."""
        results = self._collection.query(  # type: ignore[union-attr]
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self._collection.count()),  # type: ignore[union-attr]
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            strict=False,
        ):
            output.append({"text": doc, "score": 1.0 - dist, **meta})
        return output

    def _query_faiss(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        """Query FAISS index for similar chunks."""
        import faiss

        q = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)
        scores, indices = self._faiss_index.search(q, top_k)  # type: ignore[union-attr]
        output = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < len(self._faiss_docs):
                chunk = self._faiss_docs[idx].copy()
                chunk["score"] = float(score)
                output.append(chunk)
        return output

    def count(self) -> int:
        """Return the number of documents in the store."""
        self._ensure_initialised()
        if self.store_type == "chroma":
            return self._collection.count()  # type: ignore[union-attr]
        return len(self._faiss_docs)
