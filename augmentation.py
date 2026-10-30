from config import gemini_llm
from system_prompt import AUGMENTATION_SYSTEM_PROMPT


def augment_query(query: str) -> list[str] | str:
    if not query:
        print("Please enter a question.")
        return ""
    augmented_response = gemini_llm(
        query,
        system_prompt=AUGMENTATION_SYSTEM_PROMPT,
    )
    if not augmented_response:
        return [query]
    search_queries = [
        line.strip()
        for line in augmented_response.split("\n")
        if line.strip()
    ]
    if len(search_queries) < 3:
        return [query]
    return search_queries
