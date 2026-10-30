from concurrent.futures import ThreadPoolExecutor, as_completed

from augmentation import augment_query
from config import gemini_llm
from reranker import rerank_chunks
from retrieval import retrieve
from system_prompt import RETRIEVAL_SYSTEM_PROMPT


def build_prompt(query: str, hits: list[dict]) -> str:
    return (
        f"Retrieved context:\n{hits}\n\n"
        f"User question:\n{query}"
    )


def main() -> None:
    query = input("Question: ").strip()
    if not query:
        print("Please enter a question.")
        return

    augmented_queries = augment_query(query)

    retrieved_hits: list[dict] = []
    seen_chunk_ids: set[str] = set()

    with ThreadPoolExecutor(max_workers=len(augmented_queries)) as executor:
        retrieval_futures = [
            executor.submit(retrieve, search_query)
            for search_query in augmented_queries
        ]
        for future in as_completed(retrieval_futures):
            try:
                hits = future.result()
                if hits:
                    for hit in hits:
                        chunk_id = hit["id"]
                        if chunk_id in seen_chunk_ids:
                            continue
                        seen_chunk_ids.add(chunk_id)
                        retrieved_hits.append(hit)
            except Exception as error:
                print(f"⚠️ Retrieval failed for a query: {error}")

    context_hit_limit = 15
    final_hits = retrieved_hits[:context_hit_limit]

    if not final_hits:
        print("No results.")
        return

    ranked_hits = rerank_chunks(query=query, chunks=final_hits)
    if not ranked_hits:
        print("No results.")
        return

    prompt = build_prompt(query, ranked_hits)
    response = gemini_llm(query=prompt, system_prompt=RETRIEVAL_SYSTEM_PROMPT)

    print(response)


if __name__ == "__main__":
    main()