"""RAG pipeline for local, document-grounded question answering."""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ModuleNotFoundError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_MESSAGE = "This information is not available in the document."

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3
CANDIDATE_K = 6
MAX_CONTEXT_CHARS = 1800

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


SYSTEM_PROMPT = f"""
You are a document-grounded assistant.
Answer using only the context below. If the answer can be derived from the context,
respond directly and succinctly.
If the answer is not contained in the provided context, respond exactly with:
{FALLBACK_MESSAGE}
Do not use prior knowledge and do not invent facts.
""".strip()


@dataclass(frozen=True)
class DocumentStats:
    """Statistics describing the currently ingested document set."""

    page_count: int
    chunk_count: int
    character_count: int
    source_count: int


@dataclass(frozen=True)
class SourceChunk:
    """Retrieved context chunk plus retrieval scoring details."""

    content: str
    metadata: dict[str, Any]
    distance: float
    confidence: float
    vector_confidence: float
    keyword_score: float


@dataclass(frozen=True)
class AnswerResult:
    """Complete answer payload returned to the Streamlit UI."""

    answer: str
    source_chunks: list[SourceChunk]
    confidence: float
    used_llm: bool
    fallback_reason: str | None = None


class DocumentQAPipeline:
    """Build and query a FAISS-backed RAG pipeline using local Ollama."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        top_k: int = TOP_K,
        candidate_k: int = CANDIDATE_K,
        score_threshold: float = 0.0,
        max_context_chars: int = MAX_CONTEXT_CHARS,
        ollama_base_url: str | None = None,
        embeddings: HuggingFaceEmbeddings | None = None,
        llm: ChatOllama | None = None,
    ) -> None:
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.top_k = top_k
        self.candidate_k = max(candidate_k, top_k)
        self.score_threshold = score_threshold
        self.max_context_chars = max_context_chars
        self.vector_store: FAISS | None = None
        self.stats: DocumentStats | None = None

        self.embeddings = embeddings or create_embeddings(embedding_model_name)
        self.llm = llm or create_llm(model_name, ollama_base_url)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "Context:\n{context}\n\n"
                    "Question:\n{question}\n\n"
                    "Answer:",
                ),
            ]
        )
        self.chain = self.prompt | self.llm

    @property
    def is_ready(self) -> bool:
        """Return True when the FAISS index has been built."""
        return self.vector_store is not None

    def ingest_documents(
        self,
        pdf_files: list[Any] | None = None,
        raw_text: str = "",
    ) -> DocumentStats:
        """Extract, clean, split, embed, and index uploaded content."""
        source_documents: list[Document] = []
        page_count = 0

        for pdf_file in pdf_files or []:
            pdf_documents, pdf_page_count = self._extract_pdf_documents(pdf_file)
            source_documents.extend(pdf_documents)
            page_count += pdf_page_count

        cleaned_raw_text = clean_text(raw_text)
        if cleaned_raw_text:
            source_documents.append(
                Document(
                    page_content=cleaned_raw_text,
                    metadata={"source": "Pasted text", "page": None},
                )
            )

        if not source_documents:
            raise ValueError(
                "No readable text was found. Scanned PDFs need OCR before "
                "they can be used by this app."
            )

        chunks = self._split_documents(source_documents)
        if not chunks:
            raise ValueError("Document text could not be split into chunks.")

        for index, chunk in enumerate(chunks, start=1):
            chunk.metadata["chunk_id"] = index

        self.vector_store = FAISS.from_documents(
            chunks,
            self.embeddings,
            normalize_L2=True,
        )
        self.stats = DocumentStats(
            page_count=page_count,
            chunk_count=len(chunks),
            character_count=sum(len(doc.page_content) for doc in source_documents),
            source_count=len(
                {doc.metadata.get("source", "Unknown") for doc in source_documents}
            ),
        )
        return self.stats

    def answer_question(self, question: str) -> AnswerResult:
        """Answer a question using only retrieved document context."""
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question cannot be empty.")

        source_chunks = self.retrieve(cleaned_question)
        best_confidence = source_chunks[0].confidence if source_chunks else 0.0

        if best_confidence < self.score_threshold:
            return AnswerResult(
                answer=FALLBACK_MESSAGE,
                source_chunks=source_chunks,
                confidence=best_confidence,
                used_llm=False,
                fallback_reason=(
                    "Best retrieval confidence "
                    f"({best_confidence:.0%}) is below the required threshold "
                    f"({self.score_threshold:.0%})."
                ),
            )

        context = self._format_context(source_chunks)
        try:
            response = self.chain.invoke(
                {
                    "context": context,
                    "question": cleaned_question,
                }
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not reach Ollama. Make sure Ollama is running and "
                f"the '{self.model_name}' model is available."
            ) from exc

        answer = normalize_answer(extract_response_text(response))
        return AnswerResult(
            answer=answer,
            source_chunks=source_chunks,
            confidence=best_confidence,
            used_llm=True,
        )

    def retrieve(self, question: str) -> list[SourceChunk]:
        """Return the top-k most relevant chunks with confidence scores."""
        if not self.vector_store:
            raise ValueError("Process a document before asking questions.")

        results = self.vector_store.similarity_search_with_score(
            question,
            k=self.candidate_k,
        )

        question_terms = extract_significant_terms(question)
        chunks: list[SourceChunk] = []
        for document, distance in results:
            distance_float = float(distance)
            vector_confidence = distance_to_confidence(distance_float)
            keyword_score = keyword_overlap_score(
                question_terms,
                document.page_content,
            )
            confidence = blended_confidence(vector_confidence, keyword_score)
            chunks.append(
                SourceChunk(
                    content=document.page_content,
                    metadata=dict(document.metadata),
                    distance=distance_float,
                    confidence=confidence,
                    vector_confidence=vector_confidence,
                    keyword_score=keyword_score,
                )
            )

        chunks.sort(
            key=lambda chunk: (
                chunk.confidence,
                chunk.vector_confidence,
                chunk.keyword_score,
            ),
            reverse=True,
        )
        return chunks[: self.top_k]

    def _extract_pdf_documents(self, pdf_file: Any) -> tuple[list[Document], int]:
        """Extract one LangChain Document per readable PDF page."""
        source_name = getattr(pdf_file, "name", "Uploaded PDF")

        try:
            if hasattr(pdf_file, "getvalue"):
                pdf_stream = BytesIO(pdf_file.getvalue())
            else:
                if hasattr(pdf_file, "seek"):
                    pdf_file.seek(0)
                pdf_stream = pdf_file

            reader = PdfReader(pdf_stream)
            if reader.is_encrypted:
                decrypt_result = reader.decrypt("")
                if decrypt_result == 0:
                    raise ValueError("encrypted PDFs are not supported")
        except Exception as exc:
            raise ValueError(f"Could not read PDF '{source_name}': {exc}") from exc

        documents: list[Document] = []
        for page_index, page in enumerate(reader.pages, start=1):
            try:
                page_text = clean_text(page.extract_text() or "")
            except Exception as exc:
                raise ValueError(
                    f"Could not extract text from page {page_index} of "
                    f"'{source_name}': {exc}"
                ) from exc

            if page_text:
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={"source": source_name, "page": page_index},
                    )
                )

        return documents, len(reader.pages)

    @staticmethod
    def _split_documents(documents: list[Document]) -> list[Document]:
        """Split documents with the requested recursive chunking strategy."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
        )
        return splitter.split_documents(documents)

    def _format_context(self, source_chunks: list[SourceChunk]) -> str:
        """Render retrieved chunks into a compact prompt context."""
        context_parts: list[str] = []
        used_chars = 0

        for index, chunk in enumerate(source_chunks, start=1):
            source = chunk.metadata.get("source", "Unknown source")
            page = chunk.metadata.get("page")
            chunk_id = chunk.metadata.get("chunk_id", "N/A")
            page_text = f", page {page}" if page else ""
            context_part = (
                f"[Source {index}: document chunk {chunk_id}, "
                f"{source}{page_text}]\n{chunk.content}"
            )

            remaining_chars = self.max_context_chars - used_chars
            if remaining_chars <= 0:
                break
            if len(context_part) > remaining_chars:
                context_part = context_part[:remaining_chars].rsplit(" ", 1)[0]

            context_parts.append(context_part)
            used_chars += len(context_part)

        return "\n\n".join(context_parts)


def create_embeddings(
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> HuggingFaceEmbeddings:
    """Create normalized sentence-transformer embeddings."""
    try:
        return HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
    except MemoryError as exc:
        raise RuntimeError(
            "Failed to load embedding model due to insufficient memory. "
            "Increase your system page file or use a machine with more RAM, "
            "or set a smaller embedding model in rag_pipeline.DEFAULT_EMBEDDING_MODEL."
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Could not initialize embeddings: {exc}"
        ) from exc


def create_llm(
    model_name: str = DEFAULT_MODEL,
    ollama_base_url: str | None = None,
) -> ChatOllama:
    """Create an Ollama chat model with bounded output for faster answers."""
    llm_kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": 0,
        "num_ctx": 2048,
        "num_predict": 120,
        "keep_alive": "15m",
    }
    if ollama_base_url:
        llm_kwargs["base_url"] = ollama_base_url

    return ChatOllama(**llm_kwargs)


def clean_text(text: str) -> str:
    """Remove unnecessary whitespace while preserving paragraph breaks."""
    if not text:
        return ""

    normalized = text.replace("\r", "\n")
    normalized = re.sub(r"(\w)-\n(\w)", r"\1\2", normalized)
    normalized = re.sub(r"(?<!\n)\n(?!\n)", " ", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n[ \t]+", "\n", normalized)
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def distance_to_confidence(distance: float) -> float:
    """Convert normalized FAISS squared L2 distance into cosine confidence."""
    safe_distance = max(distance, 0.0)
    confidence = 1.0 - (safe_distance / 2.0)
    return max(0.0, min(1.0, confidence))


def blended_confidence(vector_confidence: float, keyword_score: float) -> float:
    """Combine semantic similarity with exact question-term overlap."""
    confidence = (0.8 * vector_confidence) + (0.2 * keyword_score)
    return max(0.0, min(1.0, confidence))


def extract_significant_terms(text: str) -> set[str]:
    """Extract useful lowercase query terms for lightweight reranking."""
    terms = set()
    for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{1,}", text.lower()):
        if term not in STOP_WORDS:
            terms.add(term)
    return terms


def keyword_overlap_score(question_terms: set[str], content: str) -> float:
    """Score how many important question terms appear in a chunk."""
    if not question_terms:
        return 0.0

    content_terms = extract_significant_terms(content)
    if not content_terms:
        return 0.0

    overlap = question_terms.intersection(content_terms)
    return len(overlap) / len(question_terms)


def extract_response_text(response: Any) -> str:
    """Extract plain text from a LangChain chat model response."""
    content = getattr(response, "content", response)

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(str(item["text"]))
            else:
                text_parts.append(str(item))
        return "\n".join(text_parts).strip()

    return str(content).strip()


def normalize_answer(answer: str) -> str:
    """Force refusal-like answers to match the required exact string."""
    cleaned_answer = answer.strip()
    if not cleaned_answer:
        return FALLBACK_MESSAGE

    lowered_answer = cleaned_answer.lower()
    refusal_phrases = (
        FALLBACK_MESSAGE.lower(),
        "i do not know",
        "i don't know",
        "not enough information",
        "not available in the document",
        "not available in the context",
        "not found in the context",
        "not mentioned in the context",
        "not provided in the context",
        "not stated in the context",
        "not in the document",
        "not in the context",
        "cannot answer",
        "can't answer",
        "please refer to the context",
        "context is unavailable",
        "context is missing",
    )

    if any(phrase in lowered_answer for phrase in refusal_phrases):
        return FALLBACK_MESSAGE

    # Additional heuristic patterns for rephrased refusals (models often
    # paraphrase the required refusal; catch common variants).
    refusal_patterns = [
        r"not\s+.*contain",
        r"not\s+.*available",
        r"not\s+.*in\s+the\s+(?:document|context|given context|provided context|given document)",
        r"does\s+not\s+appear\s+in\s+the\s+(?:document|context)",
        r"no\s+information\s+.*in\s+the",
    ]
    for pat in refusal_patterns:
        if re.search(pat, lowered_answer):
            return FALLBACK_MESSAGE

    return cleaned_answer
