# Yeh file financial PDFs ko process, clean text extract aur metadata chunks banane ka kaam karti hai.
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

# Unique hash ids generate karne ke liye md5 support module.
import hashlib
# Memory block streams stream convert karne ke liye io module.
import io
# Operating system paths aur environments functions access karne ke liye os.
import os
# Regular expression search matches patterns processing.
import re
# Formatting timestamps track dates properties.
from datetime import datetime
# Python typing supports.
from typing import Dict, List, Optional

# PDF parser toolkit plumber framework library.
import pdfplumber
# Program execution warnings logs trace parameters.
from loguru import logger


# financial documents classification keywords list check.
FINANCIAL_SECTIONS = [
    # Asset vs liabilities validation sheets.
    "balance sheet",
    # Income statements tracking checks.
    "income statement",
    # Gains losses checks.
    "profit and loss",
    # Liquid flow sheets.
    "cash flow",
    # Supplementary notes definitions section.
    "notes to accounts",
    # Audit summary records.
    "auditor's report",
    # Board summary statements.
    "chairman's statement",
    # Performance reviews.
    "management discussion",
    # Risk warnings analysis.
    "risk factors",
    # Segment wise reports.
    "segment information",
]  # Sections list end.


# PDF processing aur extraction logic control class definition.
class DocumentProcessor:
    """
    Processes financial PDFs into annotated text chunks ready for embedding.

    Pipeline:
        PDF → extract text & tables per page (OCR fallback if needed) → 
        align token stream → detect sections → sliding-window chunking → metadata
    """

    # Initialization setup values configs sets defaults.
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 100):
        """
        Args:
            chunk_size:    Target number of tokens (whitespace-split words) per chunk.
            chunk_overlap: Number of tokens to overlap between consecutive chunks.
        """
        # Chunk spacing overlap limit validation criteria rules checks.
        if chunk_overlap >= chunk_size:
            # Value boundary checks trigger standard exception.
            raise ValueError("chunk_overlap must be less than chunk_size")
        # Target token size value variable assign.
        self.chunk_size = chunk_size
        # Token overlap limit configuration parameters properties set.
        self.chunk_overlap = chunk_overlap

    # ── Public API ────────────────────────────────────────────────────────────

    # Document process full cycle pipeline execution methods definitions.
    def process_document(self, pdf_path: str, doc_name: Optional[str] = None) -> Dict:
        """
        Full pipeline: extract text/tables (OCR fallback) → detect sections → chunk → metadata.
        """
        # Name validation checking criteria fallback.
        if doc_name is None:
            # Filename parse fetch fallback configurations path parameters.
            doc_name = os.path.basename(pdf_path)

        # Stable unique identifier hash string build trigger.
        doc_id = self._generate_doc_id(doc_name)
        # Process logging info confirmations screens.
        logger.info(f"Processing document: {doc_name} (id={doc_id})")

        # 1. Extract text & tables page by page
        # File parsing stream data read logic.
        extraction = self._extract_text(pdf_path)
        # Parse status checks validation.
        if not extraction["success"]:
            # System logging prints failure trackings.
            logger.error(f"Extraction failed for {doc_name}: {extraction['error']}")
            # Empty exit status responses maps parameters returns.
            return {
                # Success flag set to False.
                "success": False,
                # Stable doc hash reference string mapping.
                "document_id": doc_id,
                # Human name configuration tag.
                "document_name": doc_name,
                # Tracking logs error string message detail mapping.
                "error": extraction["error"],
            }  # Return end.

        # Extracted pages mapping object loads reference parameters.
        pages: Dict[int, str] = extraction["pages"]
        # Total counts pages parameters mappings variables.
        total_pages = len(pages)
        # Empty inputs contents validations checks.
        if total_pages == 0:
            # Warning logger status signals.
            logger.warning(f"No pages extracted from {doc_name}")
            # Exit maps responses setup.
            return {
                # Success flag set to False.
                "success": False,
                # Unique ID map parameters.
                "document_id": doc_id,
                # File name maps.
                "document_name": doc_name,
                # Error label updates.
                "error": "No readable text or pages found in PDF.",
            }  # Return end.

        # Log completion records mappings information writes.
        logger.info(f"Extracted {total_pages} pages from {doc_name}")

        # 2. Build a global token stream with page and section metadata
        # Tokens properties items logs tracker.
        tokens_with_metadata = []
        # Page sequence iteration sorted checks.
        for page_num in sorted(pages.keys()):
            # Single page content references text fetch maps.
            page_text = pages[page_num]
            # Spacing checks validation filters configuration.
            if not page_text or not page_text.strip():
                # Skip empty pages.
                continue

            # Target category label section name evaluation logic method.
            section_name = self._detect_section(page_text)
            # Token split conversions whitespace delimiters settings.
            page_tokens = page_text.split()
            # Iteration mapping list array inserts elements mapping.
            for token in page_tokens:
                # Add item tracking tuple metrics configurations details.
                tokens_with_metadata.append((token, page_num, section_name))

        # 3. Group tokens into contiguous sections to respect section boundaries
        # Segment groups arrays.
        section_groups = []
        # Current group items list tracker placeholder.
        current_group = []
        # Current active category tracker checks status initialization value.
        current_section = None

        # Iterate sequences arrays checks metadata mapping variables.
        for token, page_num, sec_name in tokens_with_metadata:
            # Initial item boundary checking values assignments.
            if current_section is None:
                # Active category tags setting parameters configurations.
                current_section = sec_name
            # Transition category checks validation filters sequence paths.
            if sec_name != current_section:
                # Previous items data sets verification logic criteria.
                if current_group:
                    # Append completed segment results parameters mapping data inside collection.
                    section_groups.append((current_section, current_group))
                # Reset collection array items tracker.
                current_group = []
                # Update current active tags.
                current_section = sec_name
            # Element append inside current lists tracking.
            current_group.append((token, page_num))

        # Flush final items collections trackers mappings check.
        if current_group:
            # Save remnants segments data sets inside collection.
            section_groups.append((current_section, current_group))

        # 4. Generate sliding-window chunks for each section
        # Chunks objects.
        all_chunks: List[Dict] = []
        # Counter chunk sequences tracks initialization.
        chunk_index = 0
        # Calculate step dynamic shift sizes.
        step = self.chunk_size - self.chunk_overlap

        # Iterate categorised collections mapping loops parameter values.
        for sec_name, token_list in section_groups:
            # Range step slide shift index iterations.
            for start in range(0, len(token_list), step):
                # End bounds calculate.
                end = start + self.chunk_size
                # Chunks token items slice filter select metrics.
                chunk_tokens_info = token_list[start:end]
                # Empty slices checks criteria check code.
                if not chunk_tokens_info:
                    # Escape shift iterations.
                    break

                # Text compile mappings join spacing separator filters.
                chunk_text = " ".join([t[0] for t in chunk_tokens_info])
                # Extract unique page numbers referenced inside tokens lists elements.
                pages_in_chunk = sorted(list(set(t[1] for t in chunk_tokens_info)))

                # Check unique page spans bounds validations mappings.
                if len(pages_in_chunk) == 1:
                    # Single page label string conversions format maps.
                    page_range_str = str(pages_in_chunk[0])
                    # Primary page.
                    primary_page = pages_in_chunk[0]
                # Multiple page spans range indicators structure details.
                else:
                    # Multi path formatting.
                    page_range_str = f"{pages_in_chunk[0]}-{pages_in_chunk[-1]}"
                    # Primary page.
                    primary_page = pages_in_chunk[0]

                # Stable chunk hash identifiers builders sequence setup function.
                chunk_id = self._generate_chunk_id(doc_id, primary_page, chunk_index)

                # Metadata specifications options key value configuration setup.
                metadata = {
                    # Parent identifier.
                    "document_id": doc_id,
                    # Target filename.
                    "document_name": doc_name,
                    # Primary page.
                    "page_number": primary_page,
                    # Full page span string reference tag.
                    "page_range": page_range_str,
                    # Section label target.
                    "section_name": sec_name,
                    # Page total limits parameter logs.
                    "total_pages": total_pages,
                    # Local index offset.
                    "chunk_index": chunk_index,
                    # Word token metrics total.
                    "token_count": len(chunk_tokens_info),
                    # Current time tracking string configurations UTC timezone format.
                    "upload_timestamp": datetime.utcnow().isoformat() + "Z",
                }  # Metadata end.

                # Append chunk dictionary setups object inside list variables.
                all_chunks.append({
                    # Identifier hashes setup.
                    "id": chunk_id,
                    # Merged string texts.
                    "text": chunk_text,
                    # Properties configs reference.
                    "metadata": metadata,
                })  # Chunks append.
                # Increment index tracking.
                chunk_index += 1

                # Loop boundary check conditions filters.
                if end >= len(token_list):
                    # Segment loop terminations escape.
                    break

        # Console alerts logging confirmations paths.
        logger.info(f"Created {len(all_chunks)} chunks for {doc_name}")
        # Success results dictionary returns structure details specifications.
        return {
            # Standard positive status check confirmations flags.
            "success": True,
            # Document unique strings.
            "document_id": doc_id,
            # String label filenames.
            "document_name": doc_name,
            # Chunks payload collections data structures references.
            "chunks": all_chunks,
            # Total counts specifications.
            "total_chunks": len(all_chunks),
            # Total page boundary logs metadata.
            "total_pages": total_pages,
        }  # Return end.

    # ── Private helpers ───────────────────────────────────────────────────────

    # PDF file read karke text aur tables clean extracts data process functions.
    def _extract_text(self, pdf_path: str) -> Dict:
        """Extract raw text and tables from each page of the PDF."""
        # File paths existence verify check criteria setup options.
        if not os.path.exists(pdf_path):
            # Error return dictionary setup config.
            return {"success": False, "error": f"File not found: {pdf_path}"}

        # Parsing execution loop wrap inside try exceptions checking blocks.
        try:
            # Map tracking outputs configurations.
            pages = {}
            # Open connections pdfplumber wrappers.
            with pdfplumber.open(pdf_path) as pdf:
                # Iterate pages lists elements count indices.
                for page_num, page in enumerate(pdf.pages, start=1):
                    # Extracted raw text clean load parameters options.
                    text = page.extract_text() or ""
                    
                    # Extract and format tables
                    # Extract tables layout structure formats array grids.
                    tables = page.extract_tables()
                    # Tables Markdown converted contents list.
                    table_texts = []
                    # Run iteration parsing individual tables inputs logs.
                    for table in tables:
                        # Markdown table conversion method call.
                        table_md = self._table_to_markdown(table)
                        # Valid formatted table text checks criteria setup.
                        if table_md:
                            # Markdown structures save parameters lists tracking.
                            table_texts.append(table_md)  # MD table inserts.
                    
                    # Fallback to OCR if extracted text is too sparse
                    # Minimum characters limit rules check validations mappings parameters.
                    if len(text.strip()) < 50:
                        # OCR processing fallback execution trigger details configurations.
                        ocr_text = self._ocr_page(pdf_path, page_num - 1)
                        # OCR text check exists parameters values.
                        if ocr_text:
                            # Reassign extracted value properties trackers.
                            text = ocr_text
                            # Fallback triggers logs console outputs.
                            logger.info(f"OCR fallback triggered for {os.path.basename(pdf_path)}: page {page_num}")
                    
                    # Combined segments compilation variables setups.
                    combined_text = text
                    # Verification table exists parameter configurations settings.
                    if table_texts:
                        # Append tables markdown texts spacing formats tags.
                        combined_text += "\n\n" + "\n\n".join(table_texts)
                        
                    # Space check verifies inputs text content validation criteria.
                    if combined_text.strip():
                        # Save parsed text maps key page index numbers properties.
                        pages[page_num] = combined_text  # Page save.

            # Return parsed maps collections updates trackers configurations options.
            return {"success": True, "pages": pages}
        # Catch extraction failures exception alerts tracks variables parameter.
        except Exception as exc:
            # Failed returns mapping.
            return {"success": False, "error": str(exc)}

    # Scanned PDF structures OCR text extraction tool execution function.
    def _ocr_page(self, pdf_path: str, page_index: int) -> str:
        """Render a PDF page to an image and run Tesseract OCR."""
        # Dynamic import packages try catch handles.
        try:
            # PyMuPDF bindings.
            import fitz  # PyMuPDF
            # OCR packages.
            import pytesseract
            # Image loader library.
            from PIL import Image
            
            # Open documents.
            doc = fitz.open(pdf_path)
            # Boundary checks verification parameters logic.
            if page_index < 0 or page_index >= len(doc):
                # Returns empty.
                return ""
            
            # Select target page.
            page = doc[page_index]
            # Pixmap render resolution configurations settings.
            pix = page.get_pixmap(dpi=150)
            # Format converts binary bytes stream png image paths.
            img_data = pix.tobytes("png")
            # Image objects load connections.
            img = Image.open(io.BytesIO(img_data))
            
            # Text conversions OCR triggers pytesseract method call.
            text = pytesseract.image_to_string(img)
            # Output strings return sequence.
            return text
        # OCR exceptions catch tracing logs console paths warnings.
        except Exception as e:
            # Console alerts logs.
            logger.error(f"OCR failed for page {page_index + 1}: {e}")
            # Returns empty string indicators parameters.
            return ""

    # Gird structure grids list markdown table formats converts.
    def _table_to_markdown(self, table: List[List[Optional[str]]]) -> str:
        """Convert a list-of-lists table grid to Markdown table format."""
        # Verification empty inputs parameters bounds.
        if not table or not table[0]:
            # Returns empty string.
            return ""
        
        # Valid items trackers lists.
        valid_rows = []
        # Row verify loops filter empty cells segments parameters.
        for row in table:
            # Any cells elements exists check validation flags.
            if any(cell is not None and cell.strip() != "" for cell in row):
                # Format cell values sanitize None references.
                valid_rows.append([cell if cell is not None else "" for cell in row])
        
        # Valid data check indicators checks loops.
        if not valid_rows:
            # Escape empty grids.
            return ""
        
        # Col length size metrics evaluate formatting.
        col_widths = [max(len(str(cell)) for cell in col) for col in zip(*valid_rows)]
        # Markdown layout text lines collections arrays.
        markdown_lines = []
        
        # Header row
        # Pick header.
        header = valid_rows[0]
        # Text string formatting joins separators margins configs.
        header_line = "| " + " | ".join(f"{str(cell):<{col_widths[i]}}" for i, cell in enumerate(header)) + " |"
        # Save header lines inside layouts logs.
        markdown_lines.append(header_line)
        
        # Separator row
        # Separate line strings build.
        separator = "| " + " | ".join("-" * w for w in col_widths) + " |"
        # Save lines.
        markdown_lines.append(separator)
        
        # Data rows
        # Loop records items collections.
        for row in valid_rows[1:]:
            # Data row formatting align settings parameter columns metrics.
            data_line = "| " + " | ".join(f"{str(cell):<{col_widths[i]}}" for i, cell in enumerate(row)) + " |"
            # Save layouts items collections.
            markdown_lines.append(data_line)  # Append data row.
            
        # Compile Markdown table layout text output format string returns.
        return "\n" + "\n".join(markdown_lines) + "\n"

    # Category matching keyword filter heuristic section detector.
    def _detect_section(self, text: str) -> str:
        """
        Heuristic section detection based on known financial keywords.
        """
        # Header segment crop parsing search bounds validations settings.
        header_text = text[:500].lower()
        # Keyword category checking iterate loops structure configurations.
        for section in FINANCIAL_SECTIONS:
            # Category search validations checks paths.
            if section in header_text:
                # Title casing strings format return mappings.
                return section.title()
        # Default category fallback details setup variables labels.
        return "General"

    # Stable document hash generators static functions parameters.
    @staticmethod
    def _generate_doc_id(doc_name: str) -> str:
        """Stable 12-char MD5 hash of the document name."""
        # Encode name text strings generate MD5 hexadecimal crop outputs 12 bounds.
        return hashlib.md5(doc_name.encode()).hexdigest()[:12]

    # Stable chunk hash identifiers static function.
    @staticmethod
    def _generate_chunk_id(doc_id: str, page_num: int, chunk_index: int) -> str:
        """Stable 16-char MD5 hash of the chunk's position."""
        # Key string configurations formats map definitions parameters.
        key = f"{doc_id}_{page_num}_{chunk_index}"
        # Encode key string evaluate MD5 hexadecimal crop 16 limit outputs parameters.
        return hashlib.md5(key.encode()).hexdigest()[:16]

    # Chunks creator method kept for backward compatibility reasons.
    def _create_chunks(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Split text into overlapping token windows.
        """
        # Word split tokenizer spacing filters configurations.
        tokens = text.split()
        # Chunks arrays trackers setups.
        chunks = []
        # Shift shift step sizes values.
        step = self.chunk_size - self.chunk_overlap
        # Index tracking loop counts variables initialization.
        chunk_index = 0

        # Run loops range steps logic configurations details maps.
        for start in range(0, len(tokens), step):
            # End boundaries values.
            end = start + self.chunk_size
            # Slice token elements select filters sequence configurations parameters.
            chunk_tokens = tokens[start:end]
            # Empty check validations indicators escapes logic.
            if not chunk_tokens:
                # Escape loops parameters.
                break

            # Join space token strings converts format details.
            chunk_text = " ".join(chunk_tokens)
            # Hash ID generate mapping parameters methods configurations calls.
            chunk_id = self._generate_chunk_id(
                metadata["document_id"], metadata["page_number"], chunk_index
            )

            # Save chunks data format key values maps setups array inserts.
            chunks.append(
                {
                    # Hash unique string.
                    "id": chunk_id,
                    # Merged string text.
                    "text": chunk_text,
                    # Nested metadata update settings.
                    "metadata": {
                        # Map inherits.
                        **metadata,
                        # Local offset tracker index.
                        "chunk_index": chunk_index,
                        # Word tokens size measurement values.
                        "token_count": len(chunk_tokens),
                    },  # Nested metadata end.
                }
            )  # Chunks append end.
            # Local index increment.
            chunk_index += 1

            # Checks limits parameters exits.
            if end >= len(tokens):
                # Escape segment ranges loops.
                break

        # Output collections arrays returns.
        return chunks
