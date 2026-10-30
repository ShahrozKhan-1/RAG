from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import math
import re
from typing import Any

from config import embedding, vector_store


_TOKEN_PATTERN = re.compile(r"[\w]+(?:[.,/%-][\w]+)*", re.UNICODE)

_BM25_K1 = 1.5
_BM25_B = 0.3
_RRF_K = 60
_DENSE_WEIGHT = 1.0
_BM25_WEIGHT = 1.0
_MAX_DENSE_DISTANCE = 1.2
_PER_TABLE_SIBLING_CAP = 6

_corpus_cache: dict[str, Any] = {
    "collection_count": None,
    "corpus": None,
}


def _get_corpus() -> dict:
    collection_count = vector_store.count()
    cached_corpus = _corpus_cache["corpus"]
    if (
        _corpus_cache["collection_count"] == collection_count
        and cached_corpus is not None
    ):
        return cached_corpus

    corpus = vector_store.get(include=["documents", "metadatas"])
    _corpus_cache["collection_count"] = collection_count
    _corpus_cache["corpus"] = corpus
    return corpus


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.casefold())


def _is_valid_chunk(hit: dict) -> bool:
    metadata = hit.get("metadata") or {}
    section_path = " ".join(metadata.get("section_path") or []).lower()
    is_excluded_section = (
        "acknowledgement" in section_path or "contents" in section_path
    )
    if is_excluded_section:
        return False

    text = (hit.get("text") or "").lower()
    is_table_of_contents = "contents" in text and "page no." in text
    return not is_table_of_contents


def _dense_retrieve(
    query: str,
    n_results: int,
    max_distance: float = _MAX_DENSE_DISTANCE,
) -> list[dict]:
    query_embedding = embedding("search_query: " + query)
    results = vector_store.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )

    hits: list[dict] = []
    for result_index, distance in enumerate(results["distances"][0]):
        if distance is not None and distance > max_distance:
            continue
        hits.append(
            {
                "id": results["ids"][0][result_index],
                "text": results["documents"][0][result_index],
                "metadata": results["metadatas"][0][result_index] or {},
                "distance": distance,
            }
        )
    return hits


def _bm25_retrieve(query: str, n_results: int) -> list[dict]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    corpus = _get_corpus()
    documents = [
        (document_id, text, metadata or {})
        for document_id, text, metadata in zip(
            corpus["ids"],
            corpus["documents"],
            corpus["metadatas"],
        )
        if isinstance(text, str)
    ]
    if not documents:
        return []

    tokenized_documents = [
        _tokenize(text)
        for _, text, _ in documents
    ]
    document_count = len(tokenized_documents)
    average_document_length = (
        sum(map(len, tokenized_documents)) / document_count or 1
    )

    document_frequency: Counter[str] = Counter()
    for document_tokens in tokenized_documents:
        document_frequency.update(set(document_tokens))

    scored_documents: list[dict] = []
    for document, document_tokens in zip(documents, tokenized_documents):
        document_id, text, metadata = document
        term_frequency = Counter(document_tokens)
        score = 0.0

        for query_token in query_tokens:
            frequency = term_frequency[query_token]
            if not frequency:
                continue

            inverse_frequency = math.log(
                1
                + (document_count - document_frequency[query_token] + 0.5)
                / (document_frequency[query_token] + 0.5)
            )
            length_normalization = _BM25_K1 * (
                1
                - _BM25_B
                + _BM25_B * len(document_tokens) / average_document_length
            )
            score += inverse_frequency * (
                frequency * (_BM25_K1 + 1)
            ) / (frequency + length_normalization)

        if score > 0:
            scored_documents.append(
                {
                    "id": document_id,
                    "text": text,
                    "metadata": metadata,
                    "bm25_score": score,
                }
            )

    scored_documents.sort(
        key=lambda hit: hit["bm25_score"],
        reverse=True,
    )
    return scored_documents[:n_results]


def _expand_table_siblings(
    hits: list[dict],
    max_total: int,
    per_table_cap: int = _PER_TABLE_SIBLING_CAP,
) -> list[dict]:
    if not hits:
        return []

    table_ids = {
        table_id
        for hit in hits
        for table_id in (hit["metadata"].get("table_ids") or [])
    }
    if not table_ids:
        return hits[:max_total]

    corpus = _get_corpus()
    siblings_by_table: dict[str, list[dict]] = {}
    for chunk_id, text, metadata in zip(
        corpus["ids"],
        corpus["documents"],
        corpus["metadatas"],
    ):
        if not isinstance(text, str):
            continue

        chunk_metadata = metadata or {}
        for table_id in chunk_metadata.get("table_ids") or []:
            if table_id not in table_ids:
                continue
            siblings_by_table.setdefault(table_id, []).append(
                {
                    "id": chunk_id,
                    "text": text,
                    "metadata": chunk_metadata,
                    "distance": None,
                    "bm25_score": 0.0,
                    "hybrid_score": 0.0,
                }
            )

    expanded_hits: list[dict] = []
    seen_chunk_ids: set[str] = set()

    for hit in hits:
        chunk_id = hit["id"]
        if chunk_id not in seen_chunk_ids:
            expanded_hits.append(hit)
            seen_chunk_ids.add(chunk_id)

        for table_id in hit["metadata"].get("table_ids") or []:
            added_siblings = 0
            siblings = sorted(
                siblings_by_table.get(table_id, []),
                key=lambda sibling: sibling["id"],
            )
            for sibling in siblings:
                if added_siblings >= per_table_cap:
                    break
                if sibling["id"] in seen_chunk_ids:
                    continue

                expanded_hits.append(sibling)
                seen_chunk_ids.add(sibling["id"])
                added_siblings += 1

        if len(expanded_hits) >= max_total:
            break

    return expanded_hits[:max_total]


def _fuse_ranked_hits(
    dense_hits: list[dict],
    bm25_hits: list[dict],
    limit: int,
) -> list[dict]:
    fused_hits: dict[str, dict] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        candidate = fused_hits.setdefault(
            hit["id"],
            {
                "id": hit["id"],
                "text": hit["text"],
                "metadata": hit["metadata"],
                "distance": hit["distance"],
                "bm25_score": 0.0,
                "hybrid_score": 0.0,
            },
        )
        candidate["hybrid_score"] += _DENSE_WEIGHT * (
            1.0 / (_RRF_K + rank)
        )

    for rank, hit in enumerate(bm25_hits, start=1):
        candidate = fused_hits.setdefault(
            hit["id"],
            {
                "id": hit["id"],
                "text": hit["text"],
                "metadata": hit["metadata"],
                "distance": None,
                "bm25_score": hit["bm25_score"],
                "hybrid_score": 0.0,
            },
        )
        candidate["bm25_score"] = hit["bm25_score"]
        candidate["hybrid_score"] += _BM25_WEIGHT * (
            1.0 / (_RRF_K + rank)
        )

    return sorted(
        fused_hits.values(),
        key=lambda hit: hit["hybrid_score"],
        reverse=True,
    )[:limit]


def retrieve(query: str, k: int = 5) -> list[dict]:
    if k < 1:
        raise ValueError("k must be at least 1")
    if not query.strip():
        return []

    collection_size = vector_store.count()
    if collection_size == 0:
        return []

    candidate_count = min(collection_size, max(k * 8, 40))
    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(
            _dense_retrieve,
            query,
            candidate_count,
        )
        bm25_future = executor.submit(
            _bm25_retrieve,
            query,
            candidate_count,
        )

        dense_hits = [
            hit
            for hit in dense_future.result()
            if _is_valid_chunk(hit)
        ]
        bm25_hits = [
            hit
            for hit in bm25_future.result()
            if _is_valid_chunk(hit)
        ]

    top_hits = _fuse_ranked_hits(dense_hits, bm25_hits, k)
    return _expand_table_siblings(
        top_hits,
        max_total=max(k * 2, 12),
        per_table_cap=_PER_TABLE_SIBLING_CAP,
    )
