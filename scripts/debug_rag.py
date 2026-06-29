import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_pipeline import DocumentQAPipeline


def main() -> None:
    """Run a quick sanity check for the document QA pipeline."""
    pipeline = DocumentQAPipeline()
    pipeline.ingest_documents(
        pdf_files=[],
        raw_text="Hello world. This document is about testing.",
    )
    result = pipeline.answer_question("What is the document about?")

    print("Answer:", repr(result.answer))
    print("Confidence:", result.confidence)
    print("Used LLM:", result.used_llm)
    print("Fallback reason:", result.fallback_reason)
    print(
        "Source chunks:",
        [
            {
                "metadata": c.metadata,
                "distance": c.distance,
                "confidence": c.confidence,
            }
            for c in result.source_chunks
        ],
    )


if __name__ == "__main__":
    main()
