"""Embedding generation for the RAG vector store.

Supports sentence-transformers (free, local) and OpenAI embeddings.
The default is the ``all-MiniLM-L6-v2`` model which gives a good
quality/speed tradeoff for domain-specific retrieval.
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"


class EmbeddingGenerator:
    """Generates dense vector embeddings for text chunks.

    Supports two providers:
    - ``"sentence-transformers"``: Free, local inference using the
      ``sentence-transformers`` library.
    - ``"openai"``: OpenAI ``text-embedding-3-small`` (requires API key).

    Attributes:
        provider: Embedding provider (``"sentence-transformers"`` or ``"openai"``).
        model_name: Model identifier.
        embedding_dim: Dimension of output embeddings.
        batch_size: Batch size for encoding.
    """

    def __init__(
        self,
        provider: str = "sentence-transformers",
        model_name: str | None = None,
        batch_size: int = 64,
        openai_api_key: str | None = None,
    ) -> None:
        """Initialise the embedding generator.

        Args:
            provider: Embedding provider.
            model_name: Model identifier.  Defaults to ``all-MiniLM-L6-v2``
                for sentence-transformers or ``text-embedding-3-small`` for OpenAI.
            batch_size: Number of texts to embed per batch.
            openai_api_key: OpenAI API key (reads from env if not provided).
        """
        self.provider = provider
        if model_name is None:
            model_name = (
                DEFAULT_ST_MODEL if provider == "sentence-transformers" else DEFAULT_OPENAI_MODEL
            )
        self.model_name = model_name
        self.batch_size = batch_size
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self._model = None
        self.embedding_dim: int | None = None
        logger.info("EmbeddingGenerator initialised (provider=%s, model=%s)", provider, model_name)

    def _load_model(self) -> None:
        """Lazy-load the embedding model on first use."""
        if self._model is not None:
            return
        if self.provider == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                test_emb = self._model.encode(["test"])
                self.embedding_dim = test_emb.shape[1]
                logger.info("SentenceTransformer loaded (dim=%d)", self.embedding_dim)
            except ImportError:
                logger.error("sentence-transformers not installed")
                raise
        elif self.provider == "openai":
            try:
                from openai import OpenAI

                self._model = OpenAI(api_key=self.openai_api_key)
                # OpenAI text-embedding-3-small dimension
                self.embedding_dim = 1536
                logger.info("OpenAI embedding client initialised")
            except ImportError:
                logger.error("openai package not installed")
                raise
        else:
            raise ValueError(f"Unknown embedding provider: '{self.provider}'")

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            Numpy array of shape ``(len(texts), embedding_dim)``.
        """
        self._load_model()
        if not texts:
            return np.empty((0, self.embedding_dim or 384))

        if self.provider == "sentence-transformers":
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=len(texts) > 100,
                normalize_embeddings=True,
            )
            return np.array(embeddings)  # type: ignore[return-value]

        if self.provider == "openai":
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                response = self._model.embeddings.create(  # type: ignore[union-attr]
                    model=self.model_name, input=batch
                )
                batch_embs = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embs)
            return np.array(all_embeddings)

        raise ValueError(f"Unsupported provider: {self.provider}")

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query string.

        Args:
            query: Query text.

        Returns:
            1D numpy array of shape ``(embedding_dim,)``.
        """
        return self.embed_texts([query])[0]

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        """Generate embeddings for a list of chunk dictionaries.

        Args:
            chunks: List of chunk dicts from ``TextChunker``, each with a
                ``text`` key.

        Returns:
            Same list with an additional ``embedding`` key in each dict.
        """
        texts = [c["text"] for c in chunks]
        logger.info("Embedding %d chunks...", len(texts))
        embeddings = self.embed_texts(texts)
        for chunk, emb in zip(chunks, embeddings, strict=False):
            chunk["embedding"] = emb
        logger.info("Embedding complete (dim=%d)", embeddings.shape[1] if len(embeddings) else 0)
        return chunks
