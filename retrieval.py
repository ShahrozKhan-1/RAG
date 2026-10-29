from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import math
import re

from config import embedding, vector_Store



_TOKEN_PATTERN = re.compile(r"[\w]+(?:[.,/%-][\w]+)*", re.UNICODE)
_BM25_K1 = 1.5
_BM25_B = 0.3
_RRF_K = 60
_DENSE_WEIGHT = 0.7
_BM25_WEIGHT = 1.3


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.casefold())


def _is_valid_chunk(hit: dict) -> bool:
    metadata = hit["metadata"]
    types = metadata.get("types", [])
    if "table" not in types:
        return False
    section_path = " ".join(metadata.get("section_path", [])).lower()
    if "acknowledgement" in section_path or "contents" in section_path:
        return False
    page_start = metadata.get("page_start", 100)
    if page_start <= 10:
        return False
    text = hit.get("text", "").lower()
    if "contents" in text and "page no." in text:
        return False
    return True

def _dense_retrieve(query: str, n_results: int) -> list[dict]:
    query_embed = embedding("search_query: " + query)
    results = vector_Store.query(query_embeddings=query_embed, n_results=n_results)
    return [
        {
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]


def _bm25_retrieve(query: str, n_results: int) -> list[dict]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    corpus = vector_Store.get(include=["documents", "metadatas"])
    documents = [
        (doc_id, text, metadata or {})
        for doc_id, text, metadata in zip(
            corpus["ids"], corpus["documents"], corpus["metadatas"]
        )
        if isinstance(text, str)
    ]
    if not documents:
        return []

    tokenized = [_tokenize(text) for _, text, _ in documents]
    document_count = len(tokenized)
    average_length = sum(map(len, tokenized)) / document_count or 1
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))

    scores = []
    for (doc_id, text, metadata), tokens in zip(documents, tokenized):
        term_frequency = Counter(tokens)
        score = 0.0
        for term in query_tokens:
            frequency = term_frequency[term]
            if not frequency:
                continue
            inverse_frequency = math.log(
                1
                + (document_count - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            length_normalization = _BM25_K1 * (
                1 - _BM25_B + _BM25_B * len(tokens) / average_length
            )
            score += inverse_frequency * (
                frequency * (_BM25_K1 + 1)
            ) / (frequency + length_normalization)

        if score > 0:
            scores.append(
                {
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata,
                    "bm25_score": score,
                }
            )

    scores.sort(key=lambda hit: hit["bm25_score"], reverse=True)
    return scores[:n_results]


def _expand_table_siblings(hits: list[dict], max_total: int) -> list[dict]:
    table_ids = {
        tid for hit in hits for tid in hit["metadata"].get("table_ids", [])
    }
    if not table_ids:
        return hits[:max_total]

    # FIXED: ChromaDB silently fails with $in on array metadata. 
    # We fetch and filter in Python to guarantee it works.
    corpus = vector_Store.get(include=["documents", "metadatas"])
    
    siblings = {}
    for cid, text, meta in zip(corpus["ids"], corpus["documents"], corpus["metadatas"]):
        if not isinstance(text, str):
            continue
        # If this chunk belongs to any of the tables we care about, grab it.
        if set(meta.get("table_ids", [])) & table_ids:
            siblings[cid] = {
                "id": cid,
                "text": text,
                "metadata": meta,
                "distance": None,
                "bm25_score": 0.0,
                "hybrid_score": 0.0,
            }
            
    merged = {hit["id"]: hit for hit in hits}
    merged.update(siblings)

    # Group by table, then by chunk id so chunks stay in document order
    return sorted(
        merged.values(),
        key=lambda h: (
            (h["metadata"].get("table_ids") or [""])[0],
            h["id"],
        ),
    )[:max_total]


def retrieve(query: str, k: int = 5) -> list[dict]:
    if k < 1:
        raise ValueError("k must be at least 1")
    if not query.strip():
        return []

    collection_size = vector_Store.count()
    if collection_size == 0:
        return []

    # Bumped candidate count to 40 for better recall on 780 chunks
    candidate_count = min(collection_size, max(k * 8, 40))
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(_dense_retrieve, query, candidate_count)
        bm25_future = executor.submit(_bm25_retrieve, query, candidate_count)
        
        # Strict metadata filtering applied here
        dense_hits = [h for h in dense_future.result() if _is_valid_chunk(h)]
        bm25_hits = [h for h in bm25_future.result() if _is_valid_chunk(h)]

    fused: dict[str, dict] = {}
    
    # --- WEIGHTED RRF: Dense ---
    for rank, hit in enumerate(dense_hits, start=1):
        candidate = fused.setdefault(
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
        candidate["hybrid_score"] += _DENSE_WEIGHT * (1 / (_RRF_K + rank))

    # --- WEIGHTED RRF: BM25 ---
    for rank, hit in enumerate(bm25_hits, start=1):
        candidate = fused.setdefault(
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
        candidate["hybrid_score"] += _BM25_WEIGHT * (1 / (_RRF_K + rank))

    top_hits = sorted(
        fused.values(),
        key=lambda hit: hit["hybrid_score"],
        reverse=True,
    )[:k]

    # Expand to grab the rest of the table (e.g., c213#0 AND c213#1)
    return _expand_table_siblings(top_hits, max_total=max(k * 2, 12))





# results = vector_Store.get(include=["documents", "metadatas"])
# found = False
# for cid, text, meta in zip(results["ids"], results["documents"], results["metadatas"]):
#     # Look for the table title or specific text from Table 6.10
#     if "Province/Region-wise Advances" in text or "place of Disbursement" in text:
#         print(f"✅ Found chunk: {cid} | Page: {meta.get('page_start')}")
#         print(f"Types: {meta.get('types')} | Table IDs: {meta.get('table_ids')}")
#         print(f"Text preview: {text[:300]}...\n")
#         found = True

# if not found:
#     print("❌ Table 6.10 was NOT ingested into the database!")




# def get_chunk_by_id(chunk_id: str) -> str:
#     """Fetches the raw text of a specific chunk ID from ChromaDB."""
    
#     results = vector_Store.get(
#         ids=[chunk_id],
#         include=["documents", "metadatas"]
#     )
    
#     if not results["documents"] or not results["documents"][0]:
#         print(f"Chunk '{chunk_id}' not found.")
#         return ""
        
#     text = results["documents"][0]
#     metadata = results["metadatas"][0]
    
#     print(f"--- Chunk {chunk_id} ---")
#     print(f"Page: {metadata.get('page_start')} | Section: {metadata.get('section_path')}")
#     print("-" * 40)
#     print(text)
    
#     return text


# get_chunk_by_id("c178#0")