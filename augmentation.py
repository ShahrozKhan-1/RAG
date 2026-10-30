import json

from config import gemini_llm
from system_prompt import AUGMENTATION_SYSTEM_PROMPT


def augment_query(
    query: str,
    chat_history: list[dict[str, str]] | None = None,
) -> list[str] | str:
    if not query:
        print("Please enter a question.")
        return ""

    augmentation_input = query
    if chat_history:
        augmentation_input = json.dumps(
            {
                "query": query,
                "chat_history": chat_history,
            },
            ensure_ascii=False,
        )

    augmented_response = gemini_llm(
        augmentation_input,
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
