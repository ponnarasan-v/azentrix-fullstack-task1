import pytest

from rag_pipeline import (
    FALLBACK_MESSAGE,
    clean_text,
    distance_to_confidence,
    extract_response_text,
    extract_significant_terms,
    keyword_overlap_score,
    normalize_answer,
)


def test_clean_text_preserves_paragraphs_and_removes_hard_wraps():
    raw = "This is a line-\nbreak example.\nThis is another line.\n\nNew paragraph."
    cleaned = clean_text(raw)

    assert "linebreak example." in cleaned
    assert "This is another line." in cleaned
    assert "New paragraph." in cleaned
    assert "\n\n" in cleaned


@pytest.mark.parametrize(
    "distance,expected",
    [
        (0.0, 1.0),
        (0.5, 0.75),
        (2.0, 0.0),
        (3.0, 0.0),
    ],
)
def test_distance_to_confidence_maps_range(distance, expected):
    assert distance_to_confidence(distance) == expected


def test_extract_significant_terms_ignores_stop_words():
    terms = extract_significant_terms("What is the answer to this question?")

    assert "what" not in terms
    assert "answer" in terms
    assert "question" in terms


def test_keyword_overlap_score_returns_ratio():
    question_terms = {"answer", "question", "document"}
    content = "This document contains the answer to your question."

    assert keyword_overlap_score(question_terms, content) == pytest.approx(
        1.0, rel=1e-6
    )


def test_extract_response_text_handles_various_response_types():
    class Response:
        content = "Hello world"

    assert extract_response_text(Response()) == "Hello world"
    assert extract_response_text([{"text": "Hello"}, {"text": "world"}]) == "Hello\nworld"


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("", FALLBACK_MESSAGE),
        ("I don't know.", FALLBACK_MESSAGE),
        ("Not enough information available.", FALLBACK_MESSAGE),
        ("The answer is not in the document.", FALLBACK_MESSAGE),
        ("The provided information is not contained in the document.", FALLBACK_MESSAGE),
        ("This is the answer.", "This is the answer."),
    ],
)
def test_normalize_answer_enforces_fallback(answer, expected):
    assert normalize_answer(answer) == expected
