"""Streamlit UI for the Context-Aware Document Q&A Bot."""

from __future__ import annotations

import time

import streamlit as st

from rag_pipeline import (
    DEFAULT_MODEL,
    FALLBACK_MESSAGE,
    DocumentQAPipeline,
    create_embeddings,
    create_llm,
)


@st.cache_resource(show_spinner="Loading embedding model...")
def get_cached_embeddings():
    """Load the embedding model once per Streamlit server process."""
    return create_embeddings()


@st.cache_resource(show_spinner="Preparing Ollama model...")
def get_cached_llm(model_name: str):
    """Create and keep an Ollama client warm for repeated questions."""
    return create_llm(model_name)


def initialize_session_state() -> None:
    """Create Streamlit session keys used by the app."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "document_stats" not in st.session_state:
        st.session_state.document_stats = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def render_sidebar() -> None:
    """Render document ingestion controls and document statistics."""
    st.sidebar.header("Document Input")

    pdf_files = st.sidebar.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    pasted_text = st.sidebar.text_area(
        "Paste text",
        height=220,
        placeholder="Paste document text here...",
    )

    with st.sidebar.expander("Settings"):
        model_name = st.text_input("Ollama model", value=DEFAULT_MODEL)
        st.caption(
            "Fast default: tinyllama. Better but slower: llama3."
        )
        score_threshold = st.slider(
            "Minimum retrieval confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Questions below this confidence return the fallback answer.",
        )
        st.caption(
            "Use 0.00 while testing. Increase only if the bot answers from "
            "unrelated chunks."
        )

    if st.sidebar.button(
        "Process Document",
        type="primary",
        use_container_width=True,
    ):
        process_document(pdf_files, pasted_text, model_name, score_threshold)

    render_document_stats()

    if st.sidebar.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.sidebar.success("Chat history cleared.")


def process_document(
    pdf_files: list,
    pasted_text: str,
    model_name: str,
    score_threshold: float,
) -> None:
    """Build a fresh RAG pipeline from uploaded PDFs and pasted text."""
    if not pdf_files and not pasted_text.strip():
        st.sidebar.error("Upload a PDF or paste text before processing.")
        return

    try:
        model_name = model_name.strip() or DEFAULT_MODEL
        embeddings = get_cached_embeddings()
        llm = get_cached_llm(model_name)
        pipeline = DocumentQAPipeline(
            model_name=model_name,
            score_threshold=score_threshold,
            embeddings=embeddings,
            llm=llm,
        )

        started_at = time.perf_counter()
        with st.spinner("Processing document..."):
            stats = pipeline.ingest_documents(
                pdf_files=pdf_files,
                raw_text=pasted_text,
            )
        elapsed = time.perf_counter() - started_at

        st.session_state.pipeline = pipeline
        st.session_state.document_stats = stats
        st.session_state.chat_history = []
        st.sidebar.success(f"Document processed successfully in {elapsed:.1f}s.")
    except Exception as exc:
        st.sidebar.error(f"Document processing failed: {exc}")


def render_document_stats() -> None:
    """Display ingestion metrics after a document has been processed."""
    stats = st.session_state.document_stats
    if not stats:
        return

    st.sidebar.subheader("Document Statistics")
    col_one, col_two = st.sidebar.columns(2)
    col_one.metric("Pages", stats.page_count)
    col_two.metric("Chunks", stats.chunk_count)
    st.sidebar.metric("Characters", f"{stats.character_count:,}")
    st.sidebar.caption(f"Sources: {stats.source_count}")


def render_question_area() -> None:
    """Render question controls and the most recent answer."""
    pipeline: DocumentQAPipeline | None = st.session_state.pipeline

    if pipeline and pipeline.is_ready:
        st.success("Document ready. Ask a question.")
    else:
        st.info("Upload a PDF or paste text, then process the document.")

    with st.form("question_form", clear_on_submit=False):
        question = st.text_input(
            "Question",
            placeholder="Ask a question about the processed document...",
        )
        ask_clicked = st.form_submit_button("Ask", type="primary")

    if ask_clicked:
        answer_question(question)

    if st.session_state.chat_history:
        render_latest_answer(st.session_state.chat_history[-1])
        render_chat_history()


def answer_question(question: str) -> None:
    """Answer a user question using the active RAG pipeline."""
    pipeline: DocumentQAPipeline | None = st.session_state.pipeline

    if not pipeline or not pipeline.is_ready:
        st.error("Process a document before asking questions.")
        return

    if not question.strip():
        st.warning("Enter a question first.")
        return

    try:
        started_at = time.perf_counter()
        with st.spinner("Searching the document..."):
            result = pipeline.answer_question(question)
        elapsed = time.perf_counter() - started_at

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": result.answer,
                "confidence": result.confidence,
                "source_chunks": result.source_chunks,
                "fallback_reason": result.fallback_reason,
                "elapsed": elapsed,
            }
        )
    except Exception as exc:
        st.error(f"Question answering failed: {exc}")


def render_latest_answer(history_item: dict) -> None:
    """Display answer, confidence, and retrieved source chunks."""
    st.subheader("Answer")

    if history_item["answer"] == FALLBACK_MESSAGE:
        st.warning(history_item["answer"])
    else:
        st.markdown(history_item["answer"])

    st.caption(
        f"Confidence: {history_item['confidence']:.0%} | "
        f"Time: {history_item.get('elapsed', 0.0):.1f}s"
    )

    if history_item.get("fallback_reason"):
        st.info(history_item["fallback_reason"])

    with st.expander("Retrieved context", expanded=False):
        source_chunks = history_item.get("source_chunks") or []
        if not source_chunks:
            st.write("No source chunks were retrieved.")
            return

        for chunk in source_chunks:
            metadata = chunk.metadata
            source = metadata.get("source", "Unknown source")
            page = metadata.get("page")
            chunk_id = metadata.get("chunk_id", "N/A")
            page_text = f" | Page: {page}" if page else ""

            st.markdown(f"**Chunk {chunk_id} | {source}{page_text}**")
            st.caption(
                f"Overall: {chunk.confidence:.0%} | "
                f"Semantic: {chunk.vector_confidence:.0%} | "
                f"Keyword: {chunk.keyword_score:.0%} | "
                f"FAISS distance: {chunk.distance:.4f}"
            )
            st.write(chunk.content)
            st.divider()


def render_chat_history() -> None:
    """Display previous questions and answers."""
    st.subheader("Chat History")

    for item in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(item["question"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            st.caption(
                f"Confidence: {item['confidence']:.0%} | "
                f"Time: {item.get('elapsed', 0.0):.1f}s"
            )


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="Context-Aware Document Q&A Bot",
        layout="wide",
    )
    initialize_session_state()

    st.title("Context-Aware Document Q&A Bot")
    render_sidebar()
    render_question_area()


if __name__ == "__main__":
    main()
