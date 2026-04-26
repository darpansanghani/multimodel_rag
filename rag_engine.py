import os
import io
import shutil
import base64
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import qdrant_client

from openai import OpenAI
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.schema import TextNode, ImageNode, ImageDocument
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.nvidia_rerank import NVIDIARerank
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core import QueryBundle


import config
from pdf_parser import DoclingPDFParser

# Image extensions recognised during ingestion
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}

_CAPTION_PROMPT = (
    "You are a document-analysis assistant.  Describe this image in detail "
    "so that someone who cannot see it can fully understand its content.\n\n"
    "Cover:\n"
    "• The type of visual (photo, diagram, chart, table screenshot, etc.)\n"
    "• All readable text, labels, and numbers visible in the image\n"
    "• Spatial layout, relationships between elements, and data trends\n"
    "• Any colour-coding or legend information\n\n"
    "Be factual and concise — 3-6 sentences is ideal."
)

@dataclass
class RelevantImage:
    """A single image that was retrieved as evidence for the answer."""
    image_path: str
    page_num: int
    source_file: str
    image_kind: str        # "raster" or "rendered_chart"
    relevance_score: float
    image_b64: str         # base64-encoded bytes, ready for JSON
    mime_type: str         # e.g. "image/png"


@dataclass
class QueryResult:
    """Everything the API needs to build its response."""
    answer: str
    images: List[RelevantImage] = field(default_factory=list)


# Build a LlamaIndex-compatible QA template that injects the system prompt
TEXT_QA_TEMPLATE = PromptTemplate(
    template=(
        f"{config.SYSTEM_PROMPT}"
        "\n\n"
        "## Retrieved Context\n"
        "{context_str}"
        "\n\n"
        "## Question\n"
        "{query_str}"
        "\n\n"
        "## Answer"
    ),
    prompt_type=PromptType.QUESTION_ANSWER,
)

class MultiModalEngine:
    def __init__(self):
        self._qdrant = qdrant_client.QdrantClient(path=str(config.DB_DIR))

        print(f"Loading HF model {config.HF_EMBEDDING_MODEL} for text embeddings...")
        Settings.embed_model = HuggingFaceEmbedding(model_name=config.HF_EMBEDDING_MODEL)

        self.image_embed_model = ClipEmbedding()

        if getattr(config, "LLM", None) == "NVIDIA":
            self.nvidia_client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=getattr(config, "NVIDIA_API_KEY", "nvapi-@#@#@#$#@")
            )
            self.mm_llm = None
        else:
            self.mm_llm = OpenAIMultiModal(
                model="gpt-5.2",
                api_base=config.OPENAI_API_BASE,
                api_key="dummy",
                max_new_tokens=500,
            ) 

        self.text_store = QdrantVectorStore(
            client=self._qdrant, collection_name=config.QDRANT_TEXT_COLLECTION
        )
        self.image_store = QdrantVectorStore(
            client=self._qdrant, collection_name=config.QDRANT_IMAGE_COLLECTION
        )

        self.storage_context = StorageContext.from_defaults(
            vector_store=self.text_store,
            image_store=self.image_store,
        )

        # Extracted images from PDFs live here
        self.pdf_image_dir = Path(config.DB_DIR) / "pdf_images"
        self.pdf_image_dir.mkdir(parents=True, exist_ok=True)

        self._pdf_parser = DoclingPDFParser(
            image_output_dir=str(self.pdf_image_dir),
            images_scale=1.0,
        )

        self._query_engine = None
        self.index = None
        self._try_load_existing_index()

    # ------------------------------------------------------------------
    # Index bootstrap
    # ------------------------------------------------------------------

    def _try_load_existing_index(self):
        try:
            text_exists = self._qdrant.collection_exists(config.QDRANT_TEXT_COLLECTION)
            image_exists = self._qdrant.collection_exists(config.QDRANT_IMAGE_COLLECTION)

            if not text_exists:
                # Nothing ingested at all, clean slate
                print("[index] No collections found. Ingest documents to get started.")
                return

            if not image_exists:
                # Text was indexed but no images were found/stored in that run.
                # This is valid — load what we have, images will be added on next ingest.
                print(
                    "[index] Text collection found but no image collection yet. "
                    "Loading text-only index — images will appear after next ingestion."
                )
                self.index = MultiModalVectorStoreIndex.from_vector_store(
                    vector_store=self.text_store,
                    image_embed_model=self.image_embed_model,
                    # intentionally not passing image_vector_store here
                )
                return

            # Both exist, full multimodal load
            self.index = MultiModalVectorStoreIndex.from_vector_store(
                vector_store=self.text_store,
                image_vector_store=self.image_store,
                image_embed_model=self.image_embed_model,
            )
            print("[index] Loaded full multimodal index from Qdrant.")

        except Exception as e:
            print(f"[index] Could not load existing index, starting fresh. Detail: {e}")

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_documents(self, data_path: str):
        """
        Walks `data_path`.
        - .pdf files   → custom PDFParser (multi-column, tables, images, charts)
        - image files  → ImageNode + AI-generated caption TextNode
        - everything else → SimpleDirectoryReader (original behaviour)
        """
        data_path = Path(data_path)
        all_text_nodes: List[TextNode] = []
        all_image_nodes: List[ImageNode] = []

        pdf_files = list(data_path.rglob("*.pdf"))
        image_files = [
            f for f in data_path.rglob("*")
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
        ]
        other_files = [
            f for f in data_path.rglob("*")
            if f.is_file()
            and f.suffix.lower() != ".pdf"
            and f.suffix.lower() not in _IMAGE_EXTS
        ]

        # ── PDFs via custom parser ─────────────────────────────────────
        for pdf_file in pdf_files:
            print(f"[PDFParser] Parsing: {pdf_file.name}")
            try:
                t_nodes, i_nodes = self._pdf_parser.parse(str(pdf_file))
                all_text_nodes.extend(t_nodes)
                all_image_nodes.extend(i_nodes)
            except Exception as exc:
                print(f"[PDFParser] Failed on {pdf_file.name}: {exc}")

        # ── Standalone image files → ImageNode ─────────────────────────
        for img_file in image_files:
            print(f"[ingest] Adding image: {img_file.name}")
            # Copy to the persistent image directory so the path survives
            dest = self.pdf_image_dir / img_file.name
            if not dest.exists():
                shutil.copy2(str(img_file), str(dest))

            all_image_nodes.append(
                ImageNode(
                    image_path=str(dest),
                    metadata={
                        "image_path":  str(dest),
                        "source":      str(img_file),
                        "file_name":   img_file.name,
                        "page_num":    1,
                        "node_type":   "image",
                        "image_kind":  "uploaded",
                    },
                )
            )

        # ── Non-PDF / non-image files via SimpleDirectoryReader ────────
        if other_files:
            print(f"Reading {len(other_files)} non-PDF file(s) via SimpleDirectoryReader...")
            reader = SimpleDirectoryReader(
                input_files=[str(f) for f in other_files]
            )
            other_docs = reader.load_data()
            for doc in other_docs:
                all_text_nodes.append(
                    TextNode(text=doc.text, metadata=doc.metadata)
                )

        # ── Generate captions for ALL images ───────────────────────────
        # This makes every image text-searchable: a companion TextNode
        # with the AI-generated caption is linked to the ImageNode.
        if all_image_nodes:
            caption_nodes = self._caption_image_nodes(all_image_nodes)
            all_text_nodes.extend(caption_nodes)
            print(
                f"[caption] Generated {len(caption_nodes)} caption text nodes "
                f"for {len(all_image_nodes)} images."
            )

        if not all_text_nodes and not all_image_nodes:
            print("No documents found to ingest!")
            return False

        print(
            f"Indexing {len(all_text_nodes)} text nodes "
            f"and {len(all_image_nodes)} image nodes..."
        )
        self._index_nodes(all_text_nodes, all_image_nodes)
        print("Ingestion complete.")
        return True

    def _index_nodes(self, text_nodes, image_nodes):
        all_nodes = text_nodes + image_nodes

        if not self.index:
            self.index = MultiModalVectorStoreIndex(
                nodes=all_nodes,
                storage_context=self.storage_context,
                image_embed_model=self.image_embed_model,
            )

            # Qdrant only creates a collection on first write.
            # If this ingestion had no images, the image collection never gets created,
            # which causes a crash on next server restart. Create it explicitly here.
            if not self._qdrant.collection_exists(config.QDRANT_IMAGE_COLLECTION):
                self._qdrant.create_collection(
                    collection_name=config.QDRANT_IMAGE_COLLECTION,
                    vectors_config=self.image_store._collection_config(),
                )
                print("[index] Created empty image collection as placeholder.")
        else:
            for node in all_nodes:
                self.index.insert_nodes([node])

    # ------------------------------------------------------------------
    # Image captioning
    # ------------------------------------------------------------------

    def _generate_caption(self, image_path: str) -> Optional[str]:
        """
        Use the vision LLM to produce a text description of a single image.
        Returns None if captioning fails (the image will still be indexed
        via CLIP embeddings, just without a searchable text caption).
        """
        from llama_index.core.llms import ImageBlock

        try:
            b64_str, mime_type = self._encode_image(image_path)
            
            if getattr(config, "LLM", None) == "NVIDIA":
                response = self.nvidia_client.chat.completions.create(
                    model=config.NVIDIA_LLM_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": _CAPTION_PROMPT},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{b64_str}"
                                    },
                                },
                            ],
                        }
                    ],
                )
                caption = response.choices[0].message.content.strip()
            else:
                data_uri = f"data:{mime_type};base64,{b64_str}"
                img_block = ImageBlock(url=data_uri, image_mimetype=mime_type)
    
                response = self.mm_llm.complete(
                    prompt=_CAPTION_PROMPT,
                    image_documents=[img_block],
                )
                caption = str(response).strip()
                
            if caption:
                return caption
        except Exception as exc:
            print(f"[caption] Failed for {Path(image_path).name}: {exc}")
        return None

    def _caption_image_nodes(
        self, image_nodes: List[ImageNode]
    ) -> List[TextNode]:
        """
        Generate a caption for every ImageNode and return companion TextNodes.
        Each TextNode is linked to its ImageNode via NodeRelationship.SOURCE
        so that retrieval of the caption also surfaces the image.
        """
        from llama_index.core.schema import NodeRelationship, RelatedNodeInfo

        caption_text_nodes: List[TextNode] = []

        for img_node in image_nodes:
            img_path = (
                img_node.metadata.get("image_path")
                or getattr(img_node, "image_path", None)
            )
            if not img_path or not Path(img_path).exists():
                continue

            caption = self._generate_caption(img_path)
            if not caption:
                continue

            file_name = img_node.metadata.get("file_name", "unknown")
            page_num  = img_node.metadata.get("page_num", 1)
            source    = img_node.metadata.get("source", "")

            caption_node = TextNode(
                text=(
                    f"[Image caption — {file_name} page {page_num}]\n"
                    f"{caption}"
                ),
                metadata={
                    "source":     source,
                    "file_name":  file_name,
                    "page_num":   page_num,
                    "node_type":  "image_caption",
                    "image_path": img_path,
                },
            )
            # Link caption → image so co-retrieval can surface the image
            caption_node.relationships[NodeRelationship.SOURCE] = (
                RelatedNodeInfo(
                    node_id=img_node.node_id,
                    metadata={"page_num": page_num, "source": source},
                )
            )
            caption_text_nodes.append(caption_node)
            print(
                f"[caption] {file_name} p{page_num}: "
                f"{caption[:80]}{'...' if len(caption) > 80 else ''}"
            )

        return caption_text_nodes

    def _rerank_and_coretrieve(self, query: str, retrieved_nodes):
        text_nodes = [n for n in retrieved_nodes if not isinstance(n.node, ImageNode)]
        image_nodes = [n for n in retrieved_nodes if isinstance(n.node, ImageNode)]

        # ── Rerank text nodes ──────────────────────────────────────────────
        reranked_text_nodes = text_nodes
        try:
            reranker = NVIDIARerank(
                model=config.NVIDIA_RERANK_MODEL,
                api_key=config.NVIDIA_API_KEY,
                top_n=config.RERANKER_TOP_N,
            )
            reranked_text_nodes = reranker.postprocess_nodes(
                text_nodes,
                query_bundle=QueryBundle(query_str=query),
            )
            print(f"[reranker] Reranked to top {len(reranked_text_nodes)} text nodes.")
        except Exception as exc:
            print(f"[reranker] Failed, falling back to original ranking. Detail: {exc}")
            reranked_text_nodes = text_nodes[:config.RERANKER_TOP_N]

        # ── Page-aware co-retrieval ────────────────────────────────────────
        page_to_score = {}
        caption_to_score = {}

        for node_with_score in reranked_text_nodes[:config.CORETRIEVAL_TOP_PAGES]:
            node_meta = node_with_score.node.metadata
            page_num = node_meta.get("page_num")
            source = node_meta.get("source")
            score = node_with_score.score or 0.0
            
            if page_num is not None:
                key = (source, page_num)
                page_to_score[key] = max(page_to_score.get(key, 0.0), score)
                print(f"[co-retrieval] Including images from: {source} page {page_num} (score: {score:.3f})")

            # If this text node IS a caption, directly surface its image
            if node_meta.get("node_type") == "image_caption":
                cap_path = node_meta.get("image_path")
                if cap_path and Path(cap_path).exists():
                    caption_to_score[cap_path] = max(caption_to_score.get(cap_path, 0.0), score)

        from llama_index.core.schema import NodeWithScore
        
        coretreived_images = []
        for n in image_nodes:
            key1 = (n.node.metadata.get("source"), n.node.metadata.get("page_num"))
            key2 = n.node.metadata.get("image_path")
            
            is_coretrieved = False
            # Boost image score using the highest text reranker score for its page/caption
            boosted_score = n.score or 0.0

            if key1 in page_to_score:
                is_coretrieved = True
                boosted_score = max(boosted_score, page_to_score[key1])
                
            if key2 in caption_to_score:
                is_coretrieved = True
                boosted_score = max(boosted_score, caption_to_score[key2])

            if is_coretrieved:
                coretreived_images.append(NodeWithScore(node=n.node, score=boosted_score))

        high_score_images = [
            n for n in image_nodes
            if n.score and n.score >= config.IMAGE_RELEVANCE_THRESHOLD
            and not any(n.node.node_id == c.node.node_id for c in coretreived_images)
        ]

        all_images = coretreived_images + high_score_images

        # ── Deduplicate by image file content ──
        # Same logo embedded across 10 pages = 1 result, not 10
        import hashlib
        from PIL import Image as PilImage
        seen_hashes = set()
        deduplicated = []
        for n in all_images:
            img_path = n.node.metadata.get("image_path", "")
            if not img_path or not Path(img_path).exists():
                continue
            
            try:
                # Hash the raw pixel data to ignore file metadata (like PNG timestamps)
                with PilImage.open(img_path) as img:
                    # Resize to a tiny thumb before hashing to catch slight resolution differences
                    thumb = img.convert("RGB").resize((64, 64), PilImage.Resampling.LANCZOS)
                    file_hash = hashlib.md5(thumb.tobytes()).hexdigest()
            except Exception as e:
                print(f"[dedup] Could not read {img_path}: {e}")
                continue

            if file_hash not in seen_hashes:
                seen_hashes.add(file_hash)
                deduplicated.append(n)

        print(
            f"[co-retrieval] {len(coretreived_images)} page-matched + "
            f"{len(high_score_images)} high-score → "
            f"{len(deduplicated)} after dedup."
        )
        return reranked_text_nodes, deduplicated

    # ── Query ──────────────────────────────────────────────────────────────────

    def retrieve_documents(self, prompt: str):
        """Retrieves and reranks documents based on the query."""
        if not self.index:
            return [], []

        print(f"[query] {prompt}")

        retriever = self.index.as_retriever(
            similarity_top_k=config.RETRIEVER_TOP_K,
            image_similarity_top_k=config.IMAGE_RETRIEVER_TOP_K,
        )
        retrieved_nodes = retriever.retrieve(prompt)

        return self._rerank_and_coretrieve(prompt, retrieved_nodes)

    def generate_rag_response(self, prompt: str, final_text_nodes, final_image_nodes, temperature: float = 0.7, max_new_tokens: int = 500) -> QueryResult:
        """Generates a response using the retrieved documents."""
        if not self.index:
            return QueryResult(
                answer="No documents have been ingested yet. Please upload files via the sidebar first."
            )

        context_str = "\n\n---\n\n".join(
            node.node.get_content() for node in final_text_nodes
        )

        filled_prompt = TEXT_QA_TEMPLATE.format(
            context_str=context_str,
            query_str=prompt,
        )

        # Build image documents only for nodes that have a readable file on disk
        valid_image_nodes = []
        for n in final_image_nodes:
            img_path = n.node.metadata.get("image_path", "")
            if img_path and Path(img_path).exists():
                valid_image_nodes.append(n)
            else:
                print(f"[query] Skipping missing image: {img_path}")

        if getattr(config, "LLM", None) == "NVIDIA":
            content_args = [{"type": "text", "text": filled_prompt}]
            for n in valid_image_nodes:
                img_path = n.node.metadata.get("image_path")
                try:
                    b64_str, mime_type = self._encode_image(img_path)
                    content_args.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{b64_str}"
                        }
                    })
                except Exception as exc:
                    print(f"[query] Could not encode image for LLM: {img_path} — {exc}")

            response = self.nvidia_client.chat.completions.create(
                model=config.NVIDIA_LLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": content_args
                    }
                ],
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            llm_response = response.choices[0].message.content
        else:
            # Build ImageBlock list — the correct type for OpenAIMultiModal.complete().
            if valid_image_nodes:
                from llama_index.core.llms import ImageBlock
    
                image_documents = []
                for n in valid_image_nodes:
                    img_path = n.node.metadata.get("image_path")
                    try:
                        b64_str, mime_type = self._encode_image(img_path)
                        data_uri = f"data:{mime_type};base64,{b64_str}"
                        image_documents.append(
                            ImageBlock(url=data_uri, image_mimetype=mime_type)
                        )
                    except Exception as exc:
                        print(f"[query] Could not encode image for LLM: {img_path} — {exc}")
    
                if image_documents:
                    llm_response = self.mm_llm.complete(
                        prompt=filled_prompt,
                        image_documents=image_documents,
                    )
                else:
                    llm_response = self.mm_llm.complete(prompt=filled_prompt, image_documents=[])
            else:
                llm_response = self.mm_llm.complete(prompt=filled_prompt, image_documents=[])

        images = self._collect_images(valid_image_nodes)
        return QueryResult(answer=str(llm_response), images=images)

    def ask_question(self, prompt: str) -> QueryResult:
        """Legacy entry point, forwards to the new split methods."""
        final_text_nodes, final_image_nodes = self.retrieve_documents(prompt)
        return self.generate_rag_response(prompt, final_text_nodes, final_image_nodes)

    def _collect_images(self, image_nodes) -> List[RelevantImage]:
        """
        Encodes final image nodes to base64.
        Deduplication already happened in _rerank_and_coretrieve,
        so we just encode and return here.
        """
        results = []

        for node_with_score in image_nodes:
            node = node_with_score.node
            score = node_with_score.score or 0.0

            image_path = node.metadata.get("image_path") or getattr(node, "image_path", None)
            if not image_path or not Path(image_path).exists():
                continue

            try:
                image_b64, mime_type = self._encode_image(image_path)
            except Exception as exc:
                print(f"[query] Could not encode image {image_path}: {exc}")
                continue

            results.append(
                RelevantImage(
                    image_path=image_path,
                    page_num=node.metadata.get("page_num", -1),
                    source_file=node.metadata.get("file_name", "unknown"),
                    image_kind=node.metadata.get("image_kind", "unknown"),
                    relevance_score=round(score, 4),
                    image_b64=image_b64,
                    mime_type=mime_type,
                )
            )

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:3]

    @staticmethod
    def _encode_image(image_path: str):
        """
        Reads an image, resizes it so its longest edge is at most IMAGE_MAX_DIMENSION,
        then returns a (base64_string, mime_type) tuple.
        Resizing happens in memory — the file on disk is never touched.
        """
        from PIL import Image as PilImage
        import io

        suffix = Path(image_path).suffix.lower()
        mime_map = {
            ".png":  "image/png",
            ".jpg":  "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif":  "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(suffix, "image/png")
        pil_format = "PNG" if mime_type == "image/png" else "JPEG"

        img = PilImage.open(image_path)

        # Only downscale — never upscale a small image
        max_dim = config.IMAGE_MAX_DIMENSION
        if max(img.width, img.height) > max_dim:
            img.thumbnail((max_dim, max_dim), PilImage.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format=pil_format)
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode("utf-8"), mime_type

# Global engine instance used by the API
engine = MultiModalEngine()