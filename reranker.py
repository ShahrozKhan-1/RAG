from sentence_transformers import CrossEncoder


_RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
_reranker = CrossEncoder(_RERANKER_MODEL_NAME)


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
    scores = _reranker.predict(query_chunk_pairs)

    ranked_chunks = [
        {**chunk, "rerank_score": float(score)}
        for chunk, score in zip(chunks_with_text, scores)
    ]
    ranked_chunks.sort(
        key=lambda chunk: chunk["rerank_score"],
        reverse=True,
    )
    return ranked_chunks[:top_k]