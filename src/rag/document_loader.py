"""PDF document loader for the RAG vector store.

Loads and extracts text from PDF files (energy trading books, EIA/OPEC/IEA
reports) using PyMuPDF.  Returns raw text segments ready for chunking.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads and extracts text from PDF documents.

    Supports PyMuPDF (fitz) for high-quality PDF text extraction with
    fallback to pdfplumber.

    Attributes:
        documents_dir: Path to the directory containing PDF files.
        metadata_fields: Extra metadata fields to attach to each document chunk.
    """

    def __init__(
        self,
        documents_dir: str | Path | None = None,
    ) -> None:
        """Initialise the document loader.

        Args:
            documents_dir: Path to the directory containing PDF documents.
                Defaults to ``data/documents`` relative to the project root.
        """
        if documents_dir is None:
            documents_dir = Path("data/documents")
        self.documents_dir = Path(documents_dir)
        logger.info("DocumentLoader initialised (dir=%s)", self.documents_dir)

    def load_pdf(self, file_path: str | Path) -> list[dict]:
        """Load and extract text from a single PDF file.

        Attempts PyMuPDF first, falls back to pdfplumber.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of page dictionaries, each with keys:
            ``text``, ``page_number``, ``source``, ``filename``.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error("PDF file not found: %s", path)
            return []

        pages = self._load_with_pymupdf(path)
        if not pages:
            logger.info("Falling back to pdfplumber for %s", path.name)
            pages = self._load_with_pdfplumber(path)

        logger.info("Loaded %d pages from %s", len(pages), path.name)
        return pages

    def _load_with_pymupdf(self, path: Path) -> list[dict]:
        """Extract text using PyMuPDF (fitz).

        Args:
            path: Path to the PDF file.

        Returns:
            List of page dictionaries.
        """
        try:
            import fitz  # PyMuPDF

            pages = []
            doc = fitz.open(str(path))
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if text.strip():
                    pages.append(
                        {
                            "text": text.strip(),
                            "page_number": page_num,
                            "source": str(path),
                            "filename": path.name,
                            "total_pages": doc.page_count,
                        }
                    )
            doc.close()
            return pages
        except ImportError:
            logger.warning("PyMuPDF not available")
            return []
        except Exception as exc:
            logger.error("PyMuPDF error for %s: %s", path.name, exc)
            return []

    def _load_with_pdfplumber(self, path: Path) -> list[dict]:
        """Extract text using pdfplumber (fallback).

        Args:
            path: Path to the PDF file.

        Returns:
            List of page dictionaries.
        """
        try:
            import pdfplumber

            pages = []
            with pdfplumber.open(str(path)) as pdf:
                total = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append(
                            {
                                "text": text.strip(),
                                "page_number": page_num,
                                "source": str(path),
                                "filename": path.name,
                                "total_pages": total,
                            }
                        )
            return pages
        except ImportError:
            logger.error("Neither PyMuPDF nor pdfplumber is available")
            return []
        except Exception as exc:
            logger.error("pdfplumber error for %s: %s", path.name, exc)
            return []

    def load_directory(
        self,
        directory: str | Path | None = None,
        glob_pattern: str = "**/*.pdf",
    ) -> list[dict]:
        """Load all PDF files from a directory.

        Args:
            directory: Directory to search.  Defaults to ``self.documents_dir``.
            glob_pattern: Glob pattern for matching files.

        Returns:
            Combined list of page dictionaries from all PDFs.
        """
        search_dir = Path(directory) if directory else self.documents_dir
        pdf_files = list(search_dir.glob(glob_pattern))

        if not pdf_files:
            logger.warning("No PDF files found in %s", search_dir)
            return []

        logger.info("Found %d PDF files in %s", len(pdf_files), search_dir)
        all_pages: list[dict] = []
        for pdf_path in pdf_files:
            pages = self.load_pdf(pdf_path)
            all_pages.extend(pages)

        logger.info("Total pages loaded: %d from %d files", len(all_pages), len(pdf_files))
        return all_pages
