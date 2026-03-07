"""RAG retriever with optional reranking.

Combines embedding-based semantic search from the vector store with
optional cross-encoder reranking for improved retrieval precision.
"""

from __future__ import annotations

import logging

from src.rag.embeddings import EmbeddingGenerator
from src.rag.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves relevant document chunks for a given query.

    Performs two-stage retrieval:
    1. Dense embedding search (approximate nearest neighbours)
    2. Optional cross-encoder reranking for precision

    Attributes:
        embedder: ``EmbeddingGenerator`` for query embedding.
        vector_store: ``VectorStoreManager`` for similarity search.
        top_k: Number of candidates to retrieve before reranking.
        rerank_top_k: Number of results to keep after reranking.
        reranker_model: Cross-encoder model name for reranking.
    """

    def __init__(
        self,
        embedder: EmbeddingGenerator,
        vector_store: VectorStoreManager,
        top_k: int = 10,
        rerank_top_k: int = 5,
        reranker_model: str | None = None,
    ) -> None:
        """Initialise the retriever.

        Args:
            embedder: Embedding generator instance.
            vector_store: Initialised vector store.
            top_k: Initial retrieval count (before reranking).
            rerank_top_k: Final result count after reranking.
            reranker_model: HuggingFace cross-encoder model for reranking.
                If ``None``, reranking is skipped.
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.reranker_model = reranker_model
        self._reranker = None
        logger.info(
            "Retriever initialised (top_k=%d, rerank_top_k=%d, reranker=%s)",
            top_k,
            rerank_top_k,
            reranker_model or "None",
        )

    def _load_reranker(self) -> None:
        """Lazy-load the cross-encoder reranker."""
        if self._reranker is not None or not self.reranker_model:
            return
        try:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.reranker_model)
            logger.info("Cross-encoder reranker loaded: %s", self.reranker_model)
        except (ImportError, Exception) as exc:
            logger.warning("Could not load reranker %s: %s", self.reranker_model, exc)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: dict | None = None,
    ) -> list[dict]:
        """Retrieve and optionally rerank document chunks for a query.

        Args:
            query: Natural language query string.
            top_k: Override the default number of results.
            filter_metadata: Optional metadata filter dict (source, filename, etc.).

        Returns:
            Ranked list of retrieved document chunk dictionaries.
        """
        k = top_k or self.top_k
        query_embedding = self.embedder.embed_query(query)
        candidates = self.vector_store.query(query_embedding, top_k=k)

        if not candidates:
            logger.warning("No results retrieved for query: '%s'", query[:80])
            return []

        # Apply metadata filter if specified
        if filter_metadata:
            candidates = [
                c
                for c in candidates
                if all(c.get(key) == val for key, val in filter_metadata.items())
            ]

        # Rerank if a cross-encoder is configured
        if self.reranker_model:
            self._load_reranker()
            if self._reranker is not None:
                candidates = self._rerank(query, candidates)

        results = candidates[: self.rerank_top_k]
        logger.info("Retrieved %d chunks for query '%s...'", len(results), query[:50])
        return results

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Rerank candidates using a cross-encoder.

        Args:
            query: Original query string.
            candidates: Initial retrieval results.

        Returns:
            Reranked list of candidates.
        """
        pairs = [(query, c["text"]) for c in candidates]
        try:
            scores = self._reranker.predict(pairs)  # type: ignore[union-attr]
            for chunk, score in zip(candidates, scores, strict=False):
                chunk["rerank_score"] = float(score)
            candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            logger.debug("Reranking complete for %d candidates", len(candidates))
        except Exception as exc:
            logger.warning("Reranking failed: %s", exc)
        return candidates

    def format_context(self, chunks: list[dict], max_tokens: int = 3000) -> str:
        """Format retrieved chunks into a context string for the LLM.

        Args:
            chunks: List of retrieved chunk dictionaries.
            max_tokens: Approximate maximum context length in characters.

        Returns:
            Formatted context string with source citations.
        """
        context_parts = []
        total_chars = 0
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("filename", "Unknown")
            page = chunk.get("page_number", "?")
            text = chunk.get("text", "")
            part = f"[{i}] Source: {source} (page {page})\n{text}"
            if total_chars + len(part) > max_tokens * 4:  # ~4 chars per token
                break
            context_parts.append(part)
            total_chars += len(part)
        return "\n\n---\n\n".join(context_parts)
