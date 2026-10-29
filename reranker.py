from sentence_transformers import CrossEncoder



reranker = CrossEncoder("BAAI/bge-reranker-base")


def rerank_chunks(query: str, chunks: list[str], top_k: int = 5) -> list[str]:
    if not chunks:
        return []

    pairs = [
        (query, chunk["text"])
        for chunk in chunks
    ]

    scores = reranker.predict(pairs)

    ranked = []

    for chunk, score in zip(chunks, scores):
        ranked.append({
            **chunk,
            "rerank_score": float(score)
        })

    ranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return ranked[:top_k]
