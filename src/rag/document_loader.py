"""Document loader for the RAG vector store.

Loads and extracts text from PDFs, spreadsheets (xlsx/xls/csv), and
images/plots embedded in PDFs.  PDF plots are described using GPT-4o
vision so they can be embedded alongside textual content.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# File extensions handled by each loader
PDF_EXTENSIONS = {".pdf"}
SPREADSHEET_EXTENSIONS = {".xlsx", ".xls", ".csv"}
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | SPREADSHEET_EXTENSIONS


class DocumentLoader:
    """Loads and extracts content from PDFs, spreadsheets, and PDF images.

    Supports:
    - **PDFs**: Text extraction via PyMuPDF / pdfplumber.
    - **Spreadsheets**: xlsx, xls, csv → text via pandas.
    - **PDF images/plots**: Extracted with PyMuPDF, described by GPT-4o vision.

    Attributes:
        documents_dir: Path to the directory containing source files.
        extract_images: Whether to extract and describe images from PDFs.
        openai_api_key: API key for GPT-4o vision (image descriptions).
        vision_model: OpenAI model to use for image captioning.
        min_image_size: Minimum pixel area to consider an image worth describing.
    """

    def __init__(
        self,
        documents_dir: str | Path | None = None,
        extract_images: bool = True,
        openai_api_key: str | None = None,
        vision_model: str = "gpt-4o",
        min_image_size: int = 150,
    ) -> None:
        """Initialise the document loader.

        Args:
            documents_dir: Path to the directory containing documents.
                Defaults to ``data/documents`` relative to the project root.
            extract_images: If ``True``, extract images/plots from PDFs and
                generate text descriptions using GPT-4o vision.
            openai_api_key: OpenAI API key.  Falls back to ``OPENAI_API_KEY``
                env variable.
            vision_model: OpenAI model for image description.
            min_image_size: Minimum width **and** height in pixels for an
                extracted image to be sent to the vision model.
        """
        if documents_dir is None:
            documents_dir = Path("data/documents")
        self.documents_dir = Path(documents_dir)
        self.extract_images = extract_images
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.vision_model = vision_model
        self.min_image_size = min_image_size
        logger.info("DocumentLoader initialised (dir=%s, images=%s)", self.documents_dir, extract_images)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def load_file(self, file_path: str | Path) -> list[dict]:
        """Load a single file (PDF, xlsx, xls, or csv).

        Dispatches to the appropriate loader based on file extension.

        Args:
            file_path: Path to the file.

        Returns:
            List of page/sheet dictionaries ready for chunking.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error("File not found: %s", path)
            return []

        ext = path.suffix.lower()
        if ext in PDF_EXTENSIONS:
            return self.load_pdf(path)
        if ext in SPREADSHEET_EXTENSIONS:
            return self.load_spreadsheet(path)
        logger.warning("Unsupported file type '%s': %s", ext, path.name)
        return []

    def load_pdf(self, file_path: str | Path) -> list[dict]:
        """Load and extract text (and optionally images) from a PDF.

        Attempts PyMuPDF first, falls back to pdfplumber for text.
        If :attr:`extract_images` is ``True``, also extracts images from
        each page and describes them with GPT-4o vision.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of page dictionaries, each with keys:
            ``text``, ``page_number``, ``source``, ``filename``.
            Image descriptions are appended as extra entries with
            ``content_type`` set to ``"image_description"``.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error("PDF file not found: %s", path)
            return []

        pages = self._load_with_pymupdf(path)
        if not pages:
            logger.info("Falling back to pdfplumber for %s", path.name)
            pages = self._load_with_pdfplumber(path)

        # Optionally extract and describe images / plots
        if self.extract_images:
            image_pages = self._extract_pdf_images(path)
            pages.extend(image_pages)

        logger.info("Loaded %d items from %s", len(pages), path.name)
        return pages

    def load_spreadsheet(self, file_path: str | Path) -> list[dict]:
        """Load a spreadsheet (xlsx, xls, csv) and convert to text.

        Each sheet (or the single csv table) becomes one "page" dict whose
        ``text`` contains a markdown-style table representation.

        Args:
            file_path: Path to the spreadsheet.

        Returns:
            List of page dictionaries (one per sheet).
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas is required for spreadsheet loading")
            return []

        sheets: list[dict] = []

        if ext == ".csv":
            try:
                df = pd.read_csv(path)
                text = self._dataframe_to_text(df, sheet_name=path.stem)
                if text.strip():
                    sheets.append(
                        {
                            "text": text,
                            "page_number": 1,
                            "source": str(path),
                            "filename": path.name,
                            "total_pages": 1,
                            "content_type": "spreadsheet",
                            "sheet_name": path.stem,
                        }
                    )
            except Exception as exc:
                logger.error("Failed to read CSV %s: %s", path.name, exc)
        else:
            # xlsx / xls
            try:
                xls = pd.ExcelFile(path)
                for sheet_idx, sheet_name in enumerate(xls.sheet_names, start=1):
                    try:
                        df = xls.parse(sheet_name)
                        text = self._dataframe_to_text(df, sheet_name=sheet_name)
                        if text.strip():
                            sheets.append(
                                {
                                    "text": text,
                                    "page_number": sheet_idx,
                                    "source": str(path),
                                    "filename": path.name,
                                    "total_pages": len(xls.sheet_names),
                                    "content_type": "spreadsheet",
                                    "sheet_name": sheet_name,
                                }
                            )
                    except Exception as exc:
                        logger.warning(
                            "Failed to parse sheet '%s' in %s: %s",
                            sheet_name,
                            path.name,
                            exc,
                        )
            except Exception as exc:
                logger.error("Failed to read spreadsheet %s: %s", path.name, exc)

        logger.info("Loaded %d sheets from %s", len(sheets), path.name)
        return sheets

    def load_directory(
        self,
        directory: str | Path | None = None,
    ) -> list[dict]:
        """Load all supported files from a directory.

        Recursively finds PDFs, xlsx, xls, and csv files.

        Args:
            directory: Directory to search.  Defaults to ``self.documents_dir``.

        Returns:
            Combined list of page dictionaries from all files.
        """
        search_dir = Path(directory) if directory else self.documents_dir
        all_files = [
            f
            for f in search_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not all_files:
            logger.warning("No supported files found in %s", search_dir)
            return []

        logger.info("Found %d supported files in %s", len(all_files), search_dir)
        all_pages: list[dict] = []
        for file_path in sorted(all_files):
            pages = self.load_file(file_path)
            all_pages.extend(pages)

        logger.info(
            "Total items loaded: %d from %d files", len(all_pages), len(all_files)
        )
        return all_pages

    # ------------------------------------------------------------------ #
    #  PDF text extraction
    # ------------------------------------------------------------------ #

    def _load_with_pymupdf(self, path: Path) -> list[dict]:
        """Extract text using PyMuPDF (fitz)."""
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
                            "content_type": "pdf_text",
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
        """Extract text using pdfplumber (fallback)."""
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
                                "content_type": "pdf_text",
                            }
                        )
            return pages
        except ImportError:
            logger.error("Neither PyMuPDF nor pdfplumber is available")
            return []
        except Exception as exc:
            logger.error("pdfplumber error for %s: %s", path.name, exc)
            return []

    # ------------------------------------------------------------------ #
    #  PDF image / plot extraction + GPT-4o vision description
    # ------------------------------------------------------------------ #

    def _extract_pdf_images(self, path: Path) -> list[dict]:
        """Extract images from a PDF and describe them with GPT-4o vision.

        Uses PyMuPDF to pull raster images from each page, filters out
        tiny icons/logos, and sends the remaining images to GPT-4o for a
        textual description that captures chart data, axes, trends, etc.

        Args:
            path: Path to the PDF file.

        Returns:
            List of page dictionaries with ``content_type`` =
            ``"image_description"``.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PyMuPDF required for image extraction — skipping")
            return []

        if not self.openai_api_key:
            logger.warning(
                "OPENAI_API_KEY not set — skipping image description for %s",
                path.name,
            )
            return []

        results: list[dict] = []
        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            logger.error("Cannot open PDF for image extraction %s: %s", path.name, exc)
            return []

        image_count = 0
        for page_num, page in enumerate(doc, start=1):
            image_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    if base_image is None:
                        continue
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)

                    # Skip tiny images (icons, bullets, logos)
                    if width < self.min_image_size or height < self.min_image_size:
                        continue

                    image_bytes = base_image["image"]
                    image_ext = base_image.get("ext", "png")
                    description = self._describe_image(
                        image_bytes, image_ext, path.name, page_num
                    )
                    if description:
                        image_count += 1
                        results.append(
                            {
                                "text": description,
                                "page_number": page_num,
                                "source": str(path),
                                "filename": path.name,
                                "total_pages": doc.page_count,
                                "content_type": "image_description",
                                "image_index": img_idx,
                            }
                        )
                except Exception as exc:
                    logger.debug(
                        "Skipping image xref=%d on page %d of %s: %s",
                        xref,
                        page_num,
                        path.name,
                        exc,
                    )

        doc.close()
        logger.info(
            "Extracted and described %d images from %s", image_count, path.name
        )
        return results

    def _describe_image(
        self,
        image_bytes: bytes,
        image_ext: str,
        filename: str,
        page_number: int,
    ) -> str:
        """Send a single image to GPT-4o vision and get a text description.

        The prompt instructs the model to produce a detailed description
        of charts/plots that captures data values, axis labels, trends,
        and key take-aways — maximising retrieval usefulness.

        Args:
            image_bytes: Raw image bytes.
            image_ext: Image format extension (``"png"``, ``"jpeg"``, etc.).
            filename: Source document name (for context in the prompt).
            page_number: Page number in the source PDF.

        Returns:
            Text description string, or ``""`` on failure.
        """
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("openai package not installed — cannot describe images")
            return ""

        mime_map = {"png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg"}
        mime = mime_map.get(image_ext.lower(), f"image/{image_ext}")
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime};base64,{b64}"

        prompt = (
            f"This image is from page {page_number} of the energy market document "
            f"'{filename}'. It likely contains a chart, plot, table, or diagram.\n\n"
            "Please provide a detailed textual description that captures:\n"
            "1. The type of visualisation (bar chart, line chart, table, map, etc.)\n"
            "2. Axis labels, units, and time range\n"
            "3. Key data points, values, and trends shown\n"
            "4. Any legends, annotations, or notable patterns\n"
            "5. The main insight or takeaway from this visualisation\n\n"
            "Be specific with numbers and dates where visible. "
            "Format the description as a coherent paragraph suitable for "
            "embedding in a knowledge base."
        )

        try:
            client = OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url, "detail": "high"},
                            },
                        ],
                    }
                ],
                max_tokens=600,
                temperature=0.2,
            )
            description = response.choices[0].message.content or ""
            logger.debug(
                "Described image from %s p.%d (%d chars)",
                filename,
                page_number,
                len(description),
            )
            return description.strip()
        except Exception as exc:
            logger.warning(
                "Vision API failed for image on p.%d of %s: %s",
                page_number,
                filename,
                exc,
            )
            return ""

    # ------------------------------------------------------------------ #
    #  Spreadsheet helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dataframe_to_text(df, sheet_name: str = "") -> str:
        """Convert a pandas DataFrame into a text representation.

        Produces a header line, a markdown-style table for small frames,
        or a descriptive summary + sample rows for large ones.

        Args:
            df: pandas DataFrame.
            sheet_name: Name of the sheet (used as a header).

        Returns:
            Text representation of the data.
        """
        if df.empty:
            return ""

        lines: list[str] = []
        if sheet_name:
            lines.append(f"Sheet: {sheet_name}")
            lines.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
            lines.append("")

        # Column summary
        col_types = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
        lines.append(f"Columns: {col_types}")
        lines.append("")

        # For small tables, render the full markdown table
        max_display_rows = 60
        if len(df) <= max_display_rows:
            lines.append(df.to_markdown(index=False))
        else:
            # Show first and last rows plus numeric summary
            lines.append("First 30 rows:")
            lines.append(df.head(30).to_markdown(index=False))
            lines.append("")
            lines.append("Last 10 rows:")
            lines.append(df.tail(10).to_markdown(index=False))
            lines.append("")

            # Numeric summary
            numeric_cols = df.select_dtypes(include="number")
            if not numeric_cols.empty:
                lines.append("Numeric summary:")
                lines.append(numeric_cols.describe().to_markdown())

        return "\n".join(str(line) for line in lines)
