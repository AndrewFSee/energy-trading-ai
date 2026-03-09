"""Text chunking for RAG document processing.

Splits extracted PDF text into overlapping chunks suitable for embedding
and vector store retrieval.  Uses a built-in recursive character splitter
to avoid heavy ``transformers``/``langchain`` import overhead.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split *text* recursively using the first separator that produces
    pieces shorter than *chunk_size*.  Falls back to finer separators
    when individual pieces are still too long.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep = separators[0] if separators else ""
    remaining_seps = separators[1:] if separators else []

    if sep:
        pieces = text.split(sep)
    else:
        # Character-level fallback
        pieces = list(text)

    chunks: list[str] = []
    current = ""

    for piece in pieces:
        candidate = sep.join([current, piece]) if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If the single piece is still too large, recurse with finer sep
            if len(piece) > chunk_size and remaining_seps:
                sub = _recursive_split(piece, remaining_seps, chunk_size, chunk_overlap)
                chunks.extend(sub)
            else:
                current = piece
                continue
            current = ""
    if current:
        chunks.append(current)

    # Apply overlap: prepend tail of previous chunk to create context overlap
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-chunk_overlap:]
            merged = prev_tail + sep + chunks[i]
            # Only add overlap if it doesn't blow up the chunk too much
            if len(merged) <= chunk_size * 1.5:
                overlapped.append(merged)
            else:
                overlapped.append(chunks[i])
        return overlapped

    return chunks


class TextChunker:
    """Splits document pages into overlapping text chunks for RAG.

    Uses a hierarchical separator strategy: paragraph → sentence → word level.
    Preserves document metadata (source, page number) in each chunk.

    Attributes:
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Character overlap between consecutive chunks.
        separators: Ordered list of text separators for splitting.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        """Initialise the text chunker.

        Args:
            chunk_size: Target maximum size of each text chunk in characters.
            chunk_overlap: Number of characters to overlap between chunks
                to preserve context across chunk boundaries.
            separators: Ordered list of split separators.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _split_text(self, text: str) -> list[str]:
        """Split text using the built-in recursive character splitter."""
        return _recursive_split(
            text,
            self.separators,
            self.chunk_size,
            self.chunk_overlap,
        )

    def chunk_pages(self, pages: list[dict]) -> list[dict]:
        """Chunk a list of document pages into overlapping text segments.

        Args:
            pages: List of page dictionaries from ``DocumentLoader``.
                Each must have a ``text`` key and optional metadata keys.

        Returns:
            List of chunk dictionaries, each with ``text``, ``chunk_index``,
            ``source``, ``filename``, ``page_number``, and ``char_count`` keys.
        """
        all_chunks: list[dict] = []
        for page in pages:
            text = page.get("text", "")
            if not text.strip():
                continue
            try:
                splits = self._split_text(text)
            except Exception as exc:
                logger.warning("Chunking failed for page %s: %s", page.get("page_number"), exc)
                splits = [text]  # Fall back to the whole page as one chunk

            for chunk_idx, chunk_text in enumerate(splits):
                if not chunk_text.strip():
                    continue
                chunk = {
                    "text": chunk_text.strip(),
                    "chunk_index": chunk_idx,
                    "source": page.get("source", ""),
                    "filename": page.get("filename", ""),
                    "page_number": page.get("page_number", 0),
                    "total_pages": page.get("total_pages", 0),
                    "char_count": len(chunk_text),
                }
                all_chunks.append(chunk)

        logger.info(
            "Chunked %d pages into %d text chunks (size=%d, overlap=%d)",
            len(pages),
            len(all_chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return all_chunks

    def chunk_text(self, text: str, metadata: dict | None = None) -> list[dict]:
        """Chunk a single text string.

        Args:
            text: Text to split.
            metadata: Optional metadata dict to attach to each chunk.

        Returns:
            List of chunk dictionaries.
        """
        page = {"text": text, **(metadata or {})}
        return self.chunk_pages([page])
