import io
import re
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional

import fitz  # PyMuPDF
import math
import pdfplumber
from PIL import Image, ImageChops

from llama_index.core.schema import (
    TextNode,
    ImageNode,
    NodeRelationship,
    RelatedNodeInfo,
)


class PDFParser:
    """
    Parses complex PDFs into separate TextNodes and ImageNodes.

    Handles:
      - Multi-column layouts  (spatial block sorting)
      - Tables                (pdfplumber → markdown)
      - Embedded raster imgs  (PyMuPDF image extraction)
      - Charts / vector art   (page-region rasterisation)

    Each ImageNode is linked to its source TextNode via
    NodeRelationship.SOURCE so you can always trace back to the page.
    """

    def __init__(
        self,
        image_output_dir: str,
        min_image_size: Tuple[int, int] = (200, 200),  # was 100,100 — filters small logos
        render_dpi: int = 150,
        render_full_page_for_charts: bool = True,
    ):
        """
        Args:
            image_output_dir:              Where extracted/rendered images are saved.
            min_image_size:                Skip images smaller than (w, h) — filters
                                           out bullets, icons, decorators.
            render_dpi:                    DPI used when rasterising page regions for
                                           charts/vector graphics.
            render_full_page_for_charts:   If True, pages that have no extractable
                                           raster images but contain non-trivial vector
                                           drawings are rendered as a whole-page image.
                                           Catches matplotlib-style embedded charts.
        """
        self.image_output_dir = Path(image_output_dir)
        self.image_output_dir.mkdir(parents=True, exist_ok=True)
        self.min_image_size = min_image_size
        self.render_dpi = render_dpi
        self.render_full_page_for_charts = render_full_page_for_charts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, pdf_path: str) -> Tuple[List[TextNode], List[ImageNode]]:
        pdf_path = Path(pdf_path)
        text_nodes: List[TextNode] = []
        image_nodes: List[ImageNode] = []

        fitz_doc = fitz.open(str(pdf_path))

        # Track image hashes across ALL pages — same logo on every page = one node
        seen_image_hashes: set = set()

        with pdfplumber.open(str(pdf_path)) as plumber_doc:
            for page_num, (fitz_page, plumber_page) in enumerate(
                zip(fitz_doc, plumber_doc.pages), start=1
            ):
                page_id = f"{pdf_path.stem}_page{page_num}"

                text_content = self._extract_text(fitz_page)
                tables_md = self._extract_tables(plumber_page)
                if tables_md:
                    text_content += "\n\n## Tables\n\n" + "\n\n---\n\n".join(tables_md)

                text_node: Optional[TextNode] = None
                if text_content.strip():
                    text_node = TextNode(
                        text=text_content,
                        metadata={
                            "source":    str(pdf_path),
                            "file_name": pdf_path.name,
                            "page_num":  page_num,
                            "page_id":   page_id,
                            "node_type": "text",
                        },
                    )
                    text_nodes.append(text_node)

                raster_nodes = self._extract_raster_images(
                    fitz_doc, fitz_page, page_num, page_id, pdf_path, seen_image_hashes
                )

                # Only render figure regions — never full pages
                chart_nodes = []
                if self.render_full_page_for_charts:
                    chart_nodes = self._extract_figure_regions(
                        fitz_page, page_num, page_id, pdf_path, seen_image_hashes
                    )

                # Caption-based figure extraction — catches diagrams composed
                # of mixed content (text labels + vector shapes + small images)
                # that raster / vector extractors miss.
                caption_nodes = self._extract_captioned_figures(
                    fitz_page, page_num, page_id, pdf_path, seen_image_hashes
                )

                for img_node in raster_nodes + chart_nodes + caption_nodes:
                    if text_node is not None:
                        img_node.relationships[NodeRelationship.SOURCE] = (
                            RelatedNodeInfo(
                                node_id=text_node.node_id,
                                metadata={
                                    "page_num": page_num,
                                    "source":   str(pdf_path),
                                },
                            )
                        )
                    image_nodes.append(img_node)

        fitz_doc.close()
        print(
            f"[PDFParser] {pdf_path.name}: "
            f"{len(text_nodes)} text nodes, {len(image_nodes)} image nodes"
        )
        return text_nodes, image_nodes

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    def _extract_text(self, page: fitz.Page) -> str:
        """
        Layout-aware text extraction.

        Strategy:
          1. Get all text blocks with their bounding boxes.
          2. Assign each block to a column by comparing its x-centre to
             the page midpoint.  Works for 1-col and 2-col layouts; for
             3+ columns the page is divided proportionally.
          3. Sort: column-index → top-to-bottom (y0).
        """
        blocks = page.get_text("blocks")
        # block: (x0, y0, x1, y1, text, block_no, block_type)
        text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]

        if not text_blocks:
            return ""

        page_width = page.rect.width

        # Detect likely number of columns from x-distribution
        x_centres = [(b[0] + b[2]) / 2 for b in text_blocks]
        num_cols = self._estimate_columns(x_centres, page_width)

        col_width = page_width / num_cols

        def sort_key(b: tuple) -> Tuple[int, float]:
            x_centre = (b[0] + b[2]) / 2
            col_idx = int(x_centre // col_width)
            col_idx = min(col_idx, num_cols - 1)  # clamp
            return (col_idx, b[1])  # (column, y-top)

        text_blocks.sort(key=sort_key)
        return "\n".join(b[4].strip() for b in text_blocks)

    @staticmethod
    def _estimate_columns(x_centres: List[float], page_width: float) -> int:
        """
        Heuristic: if more than 20 % of blocks cluster in the right half
        of the page, assume 2 columns; otherwise 1.
        Extend this for 3-col layouts if needed.
        """
        if not x_centres:
            return 1
        right_count = sum(1 for x in x_centres if x > page_width / 2)
        ratio = right_count / len(x_centres)
        return 2 if 0.2 <= ratio <= 0.8 else 1

    # ------------------------------------------------------------------
    # Table extraction
    # ------------------------------------------------------------------

    def _extract_tables(self, page) -> List[str]:
        """
        Uses pdfplumber (much more accurate than PyMuPDF for tables).
        Returns a list of markdown-formatted table strings.
        """
        raw_tables = page.extract_tables()
        return [self._table_to_markdown(t) for t in raw_tables if t and t[0]]

    @staticmethod
    def _table_to_markdown(table: List[List]) -> str:
        """Convert a pdfplumber 2-D list into a GFM markdown table."""
        cleaned = [
            [str(cell).strip().replace("\n", " ") if cell is not None else ""
             for cell in row]
            for row in table
        ]
        if not cleaned:
            return ""

        header, *rows = cleaned
        col_count = len(header)

        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * col_count) + " |",
        ]
        for row in rows:
            # Normalise row width
            padded = (row + [""] * col_count)[:col_count]
            lines.append("| " + " | ".join(padded) + " |")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Raster image extraction
    # ------------------------------------------------------------------

    def _extract_raster_images(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        page_num: int,
        page_id: str,
        pdf_path: Path,
        seen_image_hashes: set,
    ) -> List[ImageNode]:
        nodes = []
        seen_xrefs = set()
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height

        # Build bbox lookup — tells us how large each image actually appears on the page
        bbox_map = {}
        try:
            for info in page.get_image_info(xrefs=True):
                xref_key = info.get("xref")
                if xref_key:
                    bbox_map[xref_key] = info.get("bbox")
        except Exception:
            pass

        for img_idx, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]

            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            # ── Placement size filter (must happen before any extraction) ──────────
            # Filter by how large the image appears on the page, not its pixel size.
            # A logo stored at 600px but placed in a 20pt box should be skipped.
            bbox = bbox_map.get(xref)
            if bbox:
                bbox_w = bbox[2] - bbox[0]
                bbox_h = bbox[3] - bbox[1]
                bbox_area = bbox_w * bbox_h

                if bbox_area < page_area * 0.02:
                    print(f"[PDFParser] Skipping small placement {bbox_w:.1f}x{bbox_h:.1f}pt on page {page_num}")
                    continue

                if bbox_area > page_area * 0.85:
                    print(f"[PDFParser] Skipping full-page image on page {page_num}")
                    continue

            # ── Try PIL first, fall back to fitz pixmap rendering ─────────────────
            # PIL doesn't support JBIG2, CCITT, and other PDF-native formats.
            # When PIL fails we render the image's bbox region via fitz instead —
            # fitz is a full PDF renderer and handles every format correctly.
            pil_img = None
            used_fitz_fallback = False

            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                pil_img = Image.open(io.BytesIO(image_bytes))

                # Normalize colorspace — exotic modes render black without this
                if pil_img.mode == "CMYK":
                    pil_img = pil_img.convert("RGB")
                elif pil_img.mode not in ("RGB", "RGBA", "L"):
                    pil_img = pil_img.convert("RGB")

            except Exception:
                # PIL couldn't open this format — render via fitz using the bbox
                if bbox:
                    try:
                        render_rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                        matrix = fitz.Matrix(self.render_dpi / 72, self.render_dpi / 72)
                        pixmap = page.get_pixmap(matrix=matrix, clip=render_rect, alpha=False)
                        pil_img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                        used_fitz_fallback = True
                        print(f"[PDFParser] Used fitz fallback for xref={xref} on page {page_num}")
                    except Exception as fallback_exc:
                        print(f"[PDFParser] Both PIL and fitz failed for xref={xref} on page {page_num}: {fallback_exc}")
                        continue
                else:
                    print(f"[PDFParser] Skipping xref={xref} on page {page_num}: PIL failed and no bbox for fitz fallback")
                    continue

            w, h = pil_img.size

            # Hard cap — anything over 6MP is a scanned page, not a figure
            if w * h > 6_000_000:
                print(f"[PDFParser] Skipping oversized image {w}x{h} on page {page_num}")
                continue

            # ── Skip rotated images ────────────────────────────────────────────────
            # Rotated images produce black skewed artifacts.
            # Skip this check for fitz-rendered images since they're already
            # rendered correctly into the page coordinate space.
            if not used_fitz_fallback:
                is_rotated = False
                try:
                    for info in page.get_image_info(xrefs=True):
                        if info.get("xref") != xref:
                            continue
                        transform = info.get("transform")
                        if transform and len(transform) >= 2:
                            a, b = transform[0], transform[1]
                            angle_deg = abs(math.degrees(math.atan2(b, a))) % 90
                            if angle_deg > 10:
                                is_rotated = True
                        break
                except Exception:
                    pass

                if is_rotated:
                    print(f"[PDFParser] Skipping rotated image xref={xref} on page {page_num}")
                    continue

            # ── Save and deduplicate ───────────────────────────────────────────────
            img_buffer = io.BytesIO()
            pil_img.save(img_buffer, format="PNG")
            final_bytes = img_buffer.getvalue()

            content_hash = hashlib.md5(final_bytes).hexdigest()
            if content_hash in seen_image_hashes:
                print(f"[PDFParser] Skipping duplicate image on page {page_num}")
                continue
            seen_image_hashes.add(content_hash)

            img_path = self._save_image(
                final_bytes, "png",
                f"{pdf_path.stem}_p{page_num}_img{img_idx}",
            )

            nodes.append(
                ImageNode(
                    image_path=str(img_path),
                    metadata={
                        "image_path":  str(img_path),
                        "source":      str(pdf_path),
                        "file_name":   pdf_path.name,
                        "page_num":    page_num,
                        "page_id":     page_id,
                        "node_type":   "image",
                        "image_index": img_idx,
                        "width":       w,
                        "height":      h,
                        "image_kind":  "raster",
                    },
                )
            )

        return nodes

    # ------------------------------------------------------------------
    # Chart / vector-drawing fallback
    # ------------------------------------------------------------------

    def _extract_figure_regions(
        self,
        page: fitz.Page,
        page_num: int,
        page_id: str,
        pdf_path: Path,
        seen_image_hashes: set,
    ) -> List[ImageNode]:
        """
        Finds actual figure/chart regions on the page and renders only those
        as cropped images — never the full page.

        Strategy:
        1. Get all vector drawing bounding boxes.
        2. Cluster nearby boxes into figure regions using a proximity merge.
        3. Only keep clusters that are large enough to be real figures.
        4. Render each cluster region at the configured DPI.

        This replaces full-page rendering entirely, which was the root cause
        of entire PDF pages appearing as images in the response.
        """
        drawings = page.get_drawings()
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height

        # Collect bounding rects of drawing elements — keep thin lines (arrows,
        # connectors) because they're essential for clustering diagram elements
        # together.  Only filter truly tiny dots and page-spanning rules.
        drawing_rects = []
        for d in drawings:
            rect = d.get("rect")
            if rect is None:
                continue
            # Skip tiny dots / points (both dimensions negligible)
            if rect.width < 5 and rect.height < 5:
                continue
            # Skip full-page borders
            if (rect.width * rect.height) > page_area * 0.7:
                continue
            # Skip page-spanning horizontal / vertical rules
            if rect.width > page_rect.width * 0.8 and rect.height < 3:
                continue
            if rect.height > page_rect.height * 0.8 and rect.width < 3:
                continue
            drawing_rects.append(rect)

        if not drawing_rects:
            return []

        # Merge nearby rects into clusters — figures are groups of drawing elements.
        # We expand each rect by a margin before checking overlaps so that elements
        # with small gaps between them still get grouped together.
        # When a new rect overlaps multiple existing clusters, all of those clusters
        # are merged into one (the old code only merged with the first match, leaving
        # orphaned sub-clusters that would later be discarded as too small).
        MERGE_MARGIN = 30  # points — increased to catch spread-out diagram elements
        clusters: list = []
        for rect in drawing_rects:
            expanded = fitz.Rect(
                rect.x0 - MERGE_MARGIN, rect.y0 - MERGE_MARGIN,
                rect.x1 + MERGE_MARGIN, rect.y1 + MERGE_MARGIN,
            )
            # Find ALL clusters this rect overlaps with
            matched_indices = []
            for i, cluster in enumerate(clusters):
                if cluster.intersects(expanded):
                    matched_indices.append(i)

            if not matched_indices:
                clusters.append(fitz.Rect(rect))
            else:
                # Merge the new rect and every matched cluster into one
                new_cluster = fitz.Rect(rect)
                for i in matched_indices:
                    c = clusters[i]
                    new_cluster.x0 = min(new_cluster.x0, c.x0)
                    new_cluster.y0 = min(new_cluster.y0, c.y0)
                    new_cluster.x1 = max(new_cluster.x1, c.x1)
                    new_cluster.y1 = max(new_cluster.y1, c.y1)
                # Remove old clusters in reverse order to keep indices valid
                for i in reversed(matched_indices):
                    clusters.pop(i)
                clusters.append(new_cluster)

        nodes = []
        matrix = fitz.Matrix(self.render_dpi / 72, self.render_dpi / 72)

        for cluster_idx, cluster in enumerate(clusters):
            cluster_area = cluster.width * cluster.height

            # Skip clusters that are too small to be a real figure
            # (less than 1.5% of page area — lowered to catch smaller diagrams)
            if cluster_area < page_area * 0.015:
                continue

            # Skip clusters that are unreasonably large (likely a full-page border
            # that slipped through the earlier filter)
            if cluster_area > page_area * 0.85:
                continue

            # Add a small padding around the cluster before rendering
            PAD = 10
            render_rect = fitz.Rect(
                max(cluster.x0 - PAD, page_rect.x0),
                max(cluster.y0 - PAD, page_rect.y0),
                min(cluster.x1 + PAD, page_rect.x1),
                min(cluster.y1 + PAD, page_rect.y1),
            )

            try:
                pixmap = page.get_pixmap(matrix=matrix, clip=render_rect, alpha=False)
                pil_img = Image.open(io.BytesIO(pixmap.tobytes("png")))
                
                # Crop whitespace to allow deduplication with caption fallback
                pil_img = self._crop_whitespace(pil_img)
                
                img_buffer = io.BytesIO()
                pil_img.save(img_buffer, format="PNG")
                image_bytes = img_buffer.getvalue()

                # Deduplicate rendered regions too
                content_hash = hashlib.md5(image_bytes).hexdigest()
                if content_hash in seen_image_hashes:
                    continue
                seen_image_hashes.add(content_hash)

                img_path = self._save_image(
                    image_bytes, "png",
                    f"{pdf_path.stem}_p{page_num}_fig{cluster_idx}",
                )

                nodes.append(
                    ImageNode(
                        image_path=str(img_path),
                        metadata={
                            "image_path": str(img_path),
                            "source":     str(pdf_path),
                            "file_name":  pdf_path.name,
                            "page_num":   page_num,
                            "page_id":    page_id,
                            "node_type":  "image",
                            "image_kind": "vector_figure",
                            "dpi":        self.render_dpi,
                        },
                    )
                )
                print(f"[PDFParser] Extracted figure region {cluster_idx} from page {page_num}")

            except Exception as exc:
                print(f"[PDFParser] Failed to render figure region on page {page_num}: {exc}")

        return nodes
    # ------------------------------------------------------------------
    # Caption-based figure extraction
    # ------------------------------------------------------------------

    # Pre-compiled pattern for figure captions (class-level constant)
    _CAPTION_RE = re.compile(
        r'^\s*(Figure|Fig\.?|FIGURE|FIG\.?)\s*\d+[.:;\s]',
        re.IGNORECASE | re.MULTILINE,
    )

    def _extract_captioned_figures(
        self,
        page: fitz.Page,
        page_num: int,
        page_id: str,
        pdf_path: Path,
        seen_image_hashes: set,
    ) -> List[ImageNode]:
        """
        Locate figure captions ("Figure 1:", "Fig. 2:", …) and render the
        page region immediately above the caption as a figure image.

        This is a powerful fallback for mixed-content diagrams (vector shapes
        + text labels + thin arrows) that raster and vector extractors miss.
        """
        page_rect = page.rect

        # ── Gather text blocks with positions ─────────────────────────────
        blocks = page.get_text("blocks")
        # (x0, y0, x1, y1, text, block_no, block_type)
        text_blocks = [
            b for b in blocks if b[6] == 0 and b[4].strip()
        ]
        if not text_blocks:
            return []

        text_blocks.sort(key=lambda b: b[1])  # sort by y-top

        # ── Find caption blocks ───────────────────────────────────────────
        nodes: List[ImageNode] = []
        matrix = fitz.Matrix(self.render_dpi / 72, self.render_dpi / 72)

        for idx, block in enumerate(text_blocks):
            bx0, by0, bx1, by1, btext, _bno, _btype = block
            if not self._CAPTION_RE.search(btext):
                continue

            caption_top = by0   # top edge of the caption line

            # ── Walk upward to find the top boundary of the figure ────────
            # The figure sits between the last body-text block and the caption.
            figure_top = page_rect.y0 + 5  # default: top of page

            for prev_idx in range(idx - 1, -1, -1):
                prev = text_blocks[prev_idx]
                prev_text   = prev[4].strip()
                prev_bottom = prev[3]   # y1
                prev_width  = prev[2] - prev[0]

                # Short text near the caption → likely a diagram label
                if caption_top - prev[3] < 5 and len(prev_text) < 60:
                    continue

                # Body-text heuristic: long text spanning > 35 % of page
                is_body_text = (
                    len(prev_text) > 60
                    and prev_width > page_rect.width * 0.35
                )
                # Also treat multi-line blocks as body text
                if is_body_text or prev_text.count('\n') > 2:
                    figure_top = prev_bottom + 3
                    break

                # If there's a large vertical gap (> 60 pt) from the
                # caption, this text is probably body and the gap is the
                # figure region.
                if caption_top - prev_bottom > 60:
                    figure_top = prev_bottom + 3
                    break

            # ── Build the render rect ─────────────────────────────────────
            margin_x = 15
            figure_rect = fitz.Rect(
                page_rect.x0 + margin_x,
                figure_top,
                page_rect.x1 - margin_x,
                caption_top - 2,
            )

            if figure_rect.height < 60 or figure_rect.width < 100:
                continue

            try:
                pixmap = page.get_pixmap(
                    matrix=matrix, clip=figure_rect, alpha=False
                )
                pil_img = Image.open(io.BytesIO(pixmap.tobytes("png")))

                # Crop whitespace to allow deduplication with vector fallback
                pil_img = self._crop_whitespace(pil_img)

                # Skip if the rendered region is almost entirely blank
                if self._is_blank_image(pil_img):
                    continue

                img_buffer = io.BytesIO()
                pil_img.save(img_buffer, format="PNG")
                image_bytes = img_buffer.getvalue()

                content_hash = hashlib.md5(image_bytes).hexdigest()
                if content_hash in seen_image_hashes:
                    continue
                seen_image_hashes.add(content_hash)

                img_path = self._save_image(
                    image_bytes, "png",
                    f"{pdf_path.stem}_p{page_num}_capfig{idx}",
                )

                nodes.append(
                    ImageNode(
                        image_path=str(img_path),
                        metadata={
                            "image_path": str(img_path),
                            "source":     str(pdf_path),
                            "file_name":  pdf_path.name,
                            "page_num":   page_num,
                            "page_id":    page_id,
                            "node_type":  "image",
                            "image_kind": "captioned_figure",
                            "dpi":        self.render_dpi,
                        },
                    )
                )
                caption_preview = btext.strip()[:60]
                print(
                    f"[PDFParser] Extracted captioned figure from page {page_num}"
                    f" ({caption_preview}...)"
                )

            except Exception as exc:
                print(
                    f"[PDFParser] Failed captioned-figure extraction "
                    f"on page {page_num}: {exc}"
                )

        return nodes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_blank_image(img: Image.Image, threshold: float = 0.97) -> bool:
        """Return True if more than *threshold* of pixels are near-white."""
        grey = img.convert("L")
        pixels = grey.getdata()
        near_white = sum(1 for p in pixels if p > 240)
        return near_white / len(pixels) > threshold

    @staticmethod
    def _crop_whitespace(img: Image.Image, padding: int = 10) -> Image.Image:
        """Crops pure white margins from an image, leaving an optional padding."""
        bg = Image.new(img.mode, img.size, (255, 255, 255))
        diff = ImageChops.difference(img, bg)
        if diff.mode != "L":
            diff = diff.convert("L")
        bbox = diff.getbbox()
        if bbox:
            # bbox is (left, upper, right, lower)
            padded_bbox = (
                max(0, bbox[0] - padding),
                max(0, bbox[1] - padding),
                min(img.width, bbox[2] + padding),
                min(img.height, bbox[3] + padding)
            )
            return img.crop(padded_bbox)
        return img

    def _save_image(self, image_bytes: bytes, ext: str, stem: str) -> Path:
        """Saves image bytes to disk with a content-hash suffix to avoid dupes."""
        digest = hashlib.md5(image_bytes).hexdigest()[:8]
        img_path = self.image_output_dir / f"{stem}_{digest}.{ext}"
        if not img_path.exists():
            img_path.write_bytes(image_bytes)
        return img_path


# ======================================================================
# Docling-based PDF Parser  (replaces the manual PDFParser above)
# ======================================================================

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import PictureItem, TableItem


class DoclingPDFParser:
    """
    AI-powered PDF parser using IBM's Docling library.

    Handles all the same cases as the manual PDFParser — multi-column
    layouts, tables, embedded images, charts/vector figures — but uses
    Docling's AI layout-analysis models instead of hand-rolled heuristics.

    Public API is identical to PDFParser so it's a drop-in replacement:
        parser.parse(pdf_path) → (List[TextNode], List[ImageNode])
    """

    def __init__(
        self,
        image_output_dir: str,
        images_scale: float = 2.0,
    ):
        """
        Args:
            image_output_dir:  Where extracted images are saved.
            images_scale:      Resolution multiplier for rendered images.
                               2.0 ≈ 144 DPI (default 1.0 = 72 DPI).
        """
        self.image_output_dir = Path(image_output_dir)
        self.image_output_dir.mkdir(parents=True, exist_ok=True)

        # Configure the Docling pipeline — memory-safe settings
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = False  # Fallback PDFParser provides higher-quality images
        pipeline_options.generate_table_images = False
        pipeline_options.images_scale = images_scale
        pipeline_options.do_ocr = False           # OCR causes std::bad_alloc on constrained RAM
        pipeline_options.do_table_structure = True  # Keep table structure for markdown export
        pipeline_options.do_picture_classification = False  # Not needed, saves RAM
        pipeline_options.document_timeout = 120    # 2-min max per document — prevents server hang
        pipeline_options.layout_batch_size = 1     # Process 1 page at a time to stay within RAM
        pipeline_options.table_batch_size = 1      # Process 1 table at a time

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

    # ------------------------------------------------------------------
    # Public API  (same signature as PDFParser.parse)
    # ------------------------------------------------------------------

    def parse(self, pdf_path: str) -> Tuple[List[TextNode], List[ImageNode]]:
        pdf_path = Path(pdf_path)
        text_nodes: List[TextNode] = []
        image_nodes: List[ImageNode] = []
        seen_image_hashes: set = set()

        print(f"[DoclingParser] Converting: {pdf_path.name}")
        result = self._converter.convert(str(pdf_path))
        doc = result.document

        # ── Extract full document text and chunk it ───────────────
        markdown_text = doc.export_to_markdown()

        if markdown_text.strip():
            from llama_index.core.node_parser import SentenceSplitter
            from llama_index.core.schema import Document

            # The entire document is too large for the reranker to process as one passage.
            # We must chunk it into smaller pieces.
            doc_obj = Document(
                text=markdown_text,
                metadata={
                    "source":    str(pdf_path),
                    "file_name": pdf_path.name,
                    "page_num":  1,  # Docling markdown encompasses all pages
                    "page_id":   f"{pdf_path.stem}_full",
                    "node_type": "text",
                }
            )
            
            splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
            chunked_nodes = splitter.get_nodes_from_documents([doc_obj])
            
            # Cast BaseNode to TextNode if necessary, though they are usually TextNodes
            text_nodes.extend([n for n in chunked_nodes if isinstance(n, TextNode)])

        # ── Walk document items for images and tables ──────────────────
        picture_idx = 0
        table_idx = 0

        for element, _level in doc.iterate_items():
            # --- Figures & pictures → ImageNode -----------------------
            if isinstance(element, PictureItem):
                try:
                    pil_img = element.get_image(doc)
                    if pil_img is None:
                        continue

                    img_buffer = io.BytesIO()
                    pil_img.save(img_buffer, format="PNG")
                    img_bytes = img_buffer.getvalue()

                    # Deduplicate
                    content_hash = hashlib.md5(img_bytes).hexdigest()
                    if content_hash in seen_image_hashes:
                        continue
                    seen_image_hashes.add(content_hash)

                    img_path = self._save_image(
                        img_bytes, "png",
                        f"{pdf_path.stem}_pic{picture_idx}",
                    )
                    picture_idx += 1

                    # Try to determine the page number from Docling's provenance
                    page_num = self._get_page_num(element)

                    image_nodes.append(
                        ImageNode(
                            image_path=str(img_path),
                            metadata={
                                "image_path": str(img_path),
                                "source":     str(pdf_path),
                                "file_name":  pdf_path.name,
                                "page_num":   page_num,
                                "page_id":    f"{pdf_path.stem}_page{page_num}",
                                "node_type":  "image",
                                "image_kind": "picture",
                                "width":      pil_img.width,
                                "height":     pil_img.height,
                            },
                        )
                    )
                except Exception as exc:
                    print(f"[DoclingParser] Failed to extract picture {picture_idx}: {exc}")

            # --- Tables → save as image + append markdown to text -----
            elif isinstance(element, TableItem):
                try:
                    # Table as markdown text for searchability
                    table_df = element.export_to_dataframe(doc)
                    if table_df is not None and not table_df.empty:
                        table_md = table_df.to_markdown(index=False)
                        page_num = self._get_page_num(element)

                        text_nodes.append(
                            TextNode(
                                text=f"## Table {table_idx + 1}\n\n{table_md}",
                                metadata={
                                    "source":    str(pdf_path),
                                    "file_name": pdf_path.name,
                                    "page_num":  page_num,
                                    "page_id":   f"{pdf_path.stem}_page{page_num}",
                                    "node_type": "table",
                                },
                            )
                        )

                    # Table as image for visual retrieval
                    pil_img = element.get_image(doc)
                    if pil_img is not None:
                        img_buffer = io.BytesIO()
                        pil_img.save(img_buffer, format="PNG")
                        img_bytes = img_buffer.getvalue()

                        content_hash = hashlib.md5(img_bytes).hexdigest()
                        if content_hash not in seen_image_hashes:
                            seen_image_hashes.add(content_hash)
                            img_path = self._save_image(
                                img_bytes, "png",
                                f"{pdf_path.stem}_table{table_idx}",
                            )
                            page_num = self._get_page_num(element)

                            image_nodes.append(
                                ImageNode(
                                    image_path=str(img_path),
                                    metadata={
                                        "image_path": str(img_path),
                                        "source":     str(pdf_path),
                                        "file_name":  pdf_path.name,
                                        "page_num":   page_num,
                                        "page_id":    f"{pdf_path.stem}_page{page_num}",
                                        "node_type":  "image",
                                        "image_kind": "table",
                                        "width":      pil_img.width,
                                        "height":     pil_img.height,
                                    },
                                )
                            )
                    table_idx += 1

                except Exception as exc:
                    print(f"[DoclingParser] Failed to extract table {table_idx}: {exc}")

        # ── Fallback Visual Extraction for Complex Diagrams ────────────
        # Docling sometimes misses complex multi-part diagrams or small
        # vector charts. The old PDFParser has 3 highly-tuned heuristic
        # visual extractors that we run here to catch any missing images.
        print(f"[DoclingParser] Running visual fallback extraction for {pdf_path.name}")
        import fitz
        fallback_parser = PDFParser(
            image_output_dir=str(self.image_output_dir),
            render_dpi=200,  # 200 DPI for sharp, high-quality image extraction
        )
        
        try:
            fitz_doc = fitz.open(str(pdf_path))
            for page_num, fitz_page in enumerate(fitz_doc, start=1):
                page_id = f"{pdf_path.stem}_page{page_num}"
                
                # 1. Catch embedded rasters missed by Docling
                image_nodes.extend(
                    fallback_parser._extract_raster_images(
                        fitz_doc, fitz_page, page_num, page_id, pdf_path, seen_image_hashes
                    )
                )
                
                # 2. Catch vector charts missed by Docling
                image_nodes.extend(
                    fallback_parser._extract_figure_regions(
                        fitz_page, page_num, page_id, pdf_path, seen_image_hashes
                    )
                )
                
                # 3. Catch complex mixed-content diagrams
                # This specifically catches "Figure 5" and similar
                image_nodes.extend(
                    fallback_parser._extract_captioned_figures(
                        fitz_page, page_num, page_id, pdf_path, seen_image_hashes
                    )
                )
            fitz_doc.close()
        except Exception as exc:
            print(f"[DoclingParser] Fallback extraction failed: {exc}")

        # ── Link images to the main text node ──────────────────────────
        if text_nodes and image_nodes:
            main_text_node = text_nodes[0]
            for img_node in image_nodes:
                img_node.relationships[NodeRelationship.SOURCE] = (
                    RelatedNodeInfo(
                        node_id=main_text_node.node_id,
                        metadata={
                            "page_num": img_node.metadata.get("page_num", 1),
                            "source":   str(pdf_path),
                        },
                    )
                )

        print(
            f"[DoclingParser] {pdf_path.name}: "
            f"{len(text_nodes)} text nodes, {len(image_nodes)} image nodes"
        )
        return text_nodes, image_nodes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_page_num(element) -> int:
        """Extract page number from a Docling document element's provenance."""
        try:
            prov = getattr(element, "prov", None)
            if prov and len(prov) > 0:
                return prov[0].page_no
        except Exception:
            pass
        return 1

    def _save_image(self, image_bytes: bytes, ext: str, stem: str) -> Path:
        """Saves image bytes to disk with a content-hash suffix to avoid dupes."""
        digest = hashlib.md5(image_bytes).hexdigest()[:8]
        img_path = self.image_output_dir / f"{stem}_{digest}.{ext}"
        if not img_path.exists():
            img_path.write_bytes(image_bytes)
        return img_path