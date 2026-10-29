lazy from system_prompt import RETRIEVAL_SYSTEM_PROMPT
lazy from config import gemini_llm
lazy from retrieval import retrieve
lazy from augmentation import augment_query
from concurrent.futures import ThreadPoolExecutor, as_completed
lazy from reranker import rerank_chunks




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
    
    print("\n--- Augmented Queries ---")
    for i, q in enumerate(augmented_queries, 1):
        print(f"{i}. {q}")
    
    print("\nRetrieving in parallel...")
    
    all_hits = []
    seen_ids = set()
    
    with ThreadPoolExecutor(max_workers=len(augmented_queries)) as executor:
        future_to_query = {executor.submit(retrieve, q): q for q in augmented_queries}
        for future in as_completed(future_to_query):
            try:
                hits = future.result()
                if hits:
                    for hit in hits:
                        if hit["id"] not in seen_ids:
                            seen_ids.add(hit["id"])
                            all_hits.append(hit)
            except Exception as e:
                print(f"⚠️ Retrieval failed for a query: {e}")
                
    max_context_hits = 15
    final_hits = all_hits[:max_context_hits]
    
    print(f"Retrieved {len(final_hits)} unique chunks.")
    
    if not final_hits:
        print("No results.")
        return

    ranked_hits = rerank_chunks(query=query, chunks=final_hits)
    if not ranked_hits:
        print("No results.")
        return
    print("ranked hits: ", ranked_hits)

    print("\nAsking LLM...")
    prompt = build_prompt(query, ranked_hits) 
    # response = llm(prompt, system_prompt=RETRIEVAL_SYSTEM_PROMPT)
    response = gemini_llm(query=prompt, system_prompt=RETRIEVAL_SYSTEM_PROMPT)

    print("\n=== Answer ===")
    print(response)



if __name__ == "__main__":
    main()