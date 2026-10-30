from threading import Lock

from sentence_transformers import CrossEncoder


_RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
_reranker: CrossEncoder | None = None
_reranker_lock = Lock()


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = CrossEncoder(_RERANKER_MODEL_NAME)
    return _reranker


def rerank_chunks(query: str, chunks: list[dict], top_k: int = 8) -> list[dict]:
    if not chunks:
        return []

    chunks_with_text = [chunk for chunk in chunks if chunk.get("text")]
    if not chunks_with_text:
        return []

    query_chunk_pairs = [
        (query, chunk["text"])
        for chunk in chunks_with_text
    ]
    scores = _get_reranker().predict(query_chunk_pairs)

    ranked_chunks = [
        {**chunk, "rerank_score": float(score)}
        for chunk, score in zip(chunks_with_text, scores)
    ]
    ranked_chunks.sort(
        key=lambda chunk: chunk["rerank_score"],
        reverse=True,
    )
    return ranked_chunks[:top_k]