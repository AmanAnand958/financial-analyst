"""
Document Processor — PDF ingestion, text extraction, and section-aware chunking.

Handles:
- Text extraction from financial PDFs using pdfplumber
- OCR fallback using PyMuPDF and PyTesseract for scanned PDFs
- Financial table extraction and Markdown table formatting
- Section-aware, token-based sliding-window chunking across pages
- Metadata tagging: document_id, page_number, page_range, section_name, timestamps
- Error handling for corrupted or malformed PDFs
"""

import hashlib
import io
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

import pdfplumber
from loguru import logger


# ── Financial section keywords for tagging ──────────────────────────────────
FINANCIAL_SECTIONS = [
    "balance sheet",
    "income statement",
    "profit and loss",
    "cash flow",
    "notes to accounts",
    "auditor's report",
    "chairman's statement",
    "management discussion",
    "risk factors",
    "segment information",
]


class DocumentProcessor:
    """
    Processes financial PDFs into annotated text chunks ready for embedding.

    Pipeline:
        PDF → extract text & tables per page (OCR fallback if needed) → 
        align token stream → detect sections → sliding-window chunking → metadata
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 100):
        """
        Args:
            chunk_size:    Target number of tokens (whitespace-split words) per chunk.
            chunk_overlap: Number of tokens to overlap between consecutive chunks.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Public API ────────────────────────────────────────────────────────────

    def process_document(self, pdf_path: str, doc_name: Optional[str] = None) -> Dict:
        """
        Full pipeline: extract text/tables (OCR fallback) → detect sections → chunk → metadata.

        Args:
            pdf_path:  Absolute path to the PDF file.
            doc_name:  Human-readable name (defaults to filename).

        Returns:
            {
                "success": bool,
                "document_id": str,
                "document_name": doc_name,
                "chunks": List[Dict],
                "total_chunks": int,
                "total_pages": int,
                "error": str  # only present on failure
            }
        """
        if doc_name is None:
            doc_name = os.path.basename(pdf_path)

        doc_id = self._generate_doc_id(doc_name)
        logger.info(f"Processing document: {doc_name} (id={doc_id})")

        # 1. Extract text & tables page by page
        extraction = self._extract_text(pdf_path)
        if not extraction["success"]:
            logger.error(f"Extraction failed for {doc_name}: {extraction['error']}")
            return {
                "success": False,
                "document_id": doc_id,
                "document_name": doc_name,
                "error": extraction["error"],
            }

        pages: Dict[int, str] = extraction["pages"]
        total_pages = len(pages)
        if total_pages == 0:
            logger.warning(f"No pages extracted from {doc_name}")
            return {
                "success": False,
                "document_id": doc_id,
                "document_name": doc_name,
                "error": "No readable text or pages found in PDF.",
            }

        logger.info(f"Extracted {total_pages} pages from {doc_name}")

        # 2. Build a global token stream with page and section metadata
        tokens_with_metadata = []
        for page_num in sorted(pages.keys()):
            page_text = pages[page_num]
            if not page_text or not page_text.strip():
                continue

            section_name = self._detect_section(page_text)
            page_tokens = page_text.split()
            for token in page_tokens:
                tokens_with_metadata.append((token, page_num, section_name))

        # 3. Group tokens into contiguous sections to respect section boundaries
        section_groups = []
        current_group = []
        current_section = None

        for token, page_num, sec_name in tokens_with_metadata:
            if current_section is None:
                current_section = sec_name
            if sec_name != current_section:
                if current_group:
                    section_groups.append((current_section, current_group))
                current_group = []
                current_section = sec_name
            current_group.append((token, page_num))

        if current_group:
            section_groups.append((current_section, current_group))

        # 4. Generate sliding-window chunks for each section
        all_chunks: List[Dict] = []
        chunk_index = 0
        step = self.chunk_size - self.chunk_overlap

        for sec_name, token_list in section_groups:
            for start in range(0, len(token_list), step):
                end = start + self.chunk_size
                chunk_tokens_info = token_list[start:end]
                if not chunk_tokens_info:
                    break

                chunk_text = " ".join([t[0] for t in chunk_tokens_info])
                pages_in_chunk = sorted(list(set(t[1] for t in chunk_tokens_info)))

                if len(pages_in_chunk) == 1:
                    page_range_str = str(pages_in_chunk[0])
                    primary_page = pages_in_chunk[0]
                else:
                    page_range_str = f"{pages_in_chunk[0]}-{pages_in_chunk[-1]}"
                    primary_page = pages_in_chunk[0]

                chunk_id = self._generate_chunk_id(doc_id, primary_page, chunk_index)

                metadata = {
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "page_number": primary_page,
                    "page_range": page_range_str,
                    "section_name": sec_name,
                    "total_pages": total_pages,
                    "chunk_index": chunk_index,
                    "token_count": len(chunk_tokens_info),
                    "upload_timestamp": datetime.utcnow().isoformat() + "Z",
                }

                all_chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": metadata,
                })
                chunk_index += 1

                if end >= len(token_list):
                    break

        logger.info(f"Created {len(all_chunks)} chunks for {doc_name}")
        return {
            "success": True,
            "document_id": doc_id,
            "document_name": doc_name,
            "chunks": all_chunks,
            "total_chunks": len(all_chunks),
            "total_pages": total_pages,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_text(self, pdf_path: str) -> Dict:
        """Extract raw text and tables from each page of the PDF."""
        if not os.path.exists(pdf_path):
            return {"success": False, "error": f"File not found: {pdf_path}"}

        try:
            pages = {}
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    
                    # Extract and format tables
                    tables = page.extract_tables()
                    table_texts = []
                    for table in tables:
                        table_md = self._table_to_markdown(table)
                        if table_md:
                            table_texts.append(table_md)
                    
                    # Fallback to OCR if extracted text is too sparse
                    if len(text.strip()) < 50:
                        ocr_text = self._ocr_page(pdf_path, page_num - 1)
                        if ocr_text:
                            text = ocr_text
                            logger.info(f"OCR fallback triggered for {os.path.basename(pdf_path)}: page {page_num}")
                    
                    combined_text = text
                    if table_texts:
                        combined_text += "\n\n" + "\n\n".join(table_texts)
                        
                    if combined_text.strip():
                        pages[page_num] = combined_text

            return {"success": True, "pages": pages}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _ocr_page(self, pdf_path: str, page_index: int) -> str:
        """Render a PDF page to an image and run Tesseract OCR."""
        try:
            import fitz  # PyMuPDF
            import pytesseract
            from PIL import Image
            
            doc = fitz.open(pdf_path)
            if page_index < 0 or page_index >= len(doc):
                return ""
            
            page = doc[page_index]
            pix = page.get_pixmap(dpi=150)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            text = pytesseract.image_to_string(img)
            return text
        except Exception as e:
            logger.error(f"OCR failed for page {page_index + 1}: {e}")
            return ""

    def _table_to_markdown(self, table: List[List[Optional[str]]]) -> str:
        """Convert a list-of-lists table grid to Markdown table format."""
        if not table or not table[0]:
            return ""
        
        valid_rows = []
        for row in table:
            if any(cell is not None and cell.strip() != "" for cell in row):
                valid_rows.append([cell if cell is not None else "" for cell in row])
        
        if not valid_rows:
            return ""
        
        col_widths = [max(len(str(cell)) for cell in col) for col in zip(*valid_rows)]
        markdown_lines = []
        
        # Header row
        header = valid_rows[0]
        header_line = "| " + " | ".join(f"{str(cell):<{col_widths[i]}}" for i, cell in enumerate(header)) + " |"
        markdown_lines.append(header_line)
        
        # Separator row
        separator = "| " + " | ".join("-" * w for w in col_widths) + " |"
        markdown_lines.append(separator)
        
        # Data rows
        for row in valid_rows[1:]:
            data_line = "| " + " | ".join(f"{str(cell):<{col_widths[i]}}" for i, cell in enumerate(row)) + " |"
            markdown_lines.append(data_line)
            
        return "\n" + "\n".join(markdown_lines) + "\n"

    def _detect_section(self, text: str) -> str:
        """
        Heuristic section detection based on known financial keywords.

        Scans the first 500 characters (likely the page header area) for
        recognisable section names.

        Returns:
            Detected section name or "General" if no match found.
        """
        header_text = text[:500].lower()
        for section in FINANCIAL_SECTIONS:
            if section in header_text:
                return section.title()
        return "General"

    @staticmethod
    def _generate_doc_id(doc_name: str) -> str:
        """Stable 12-char MD5 hash of the document name."""
        return hashlib.md5(doc_name.encode()).hexdigest()[:12]

    @staticmethod
    def _generate_chunk_id(doc_id: str, page_num: int, chunk_index: int) -> str:
        """Stable 16-char MD5 hash of the chunk's position."""
        key = f"{doc_id}_{page_num}_{chunk_index}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _create_chunks(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Split text into overlapping token windows.

        This private method is kept for backward compatibility and unit tests.
        """
        tokens = text.split()
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        chunk_index = 0

        for start in range(0, len(tokens), step):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            if not chunk_tokens:
                break

            chunk_text = " ".join(chunk_tokens)
            chunk_id = self._generate_chunk_id(
                metadata["document_id"], metadata["page_number"], chunk_index
            )

            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": chunk_index,
                        "token_count": len(chunk_tokens),
                    },
                }
            )
            chunk_index += 1

            if end >= len(tokens):
                break

        return chunks

