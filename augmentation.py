from config import gemini_llm
from system_prompt import AUGMENTATION_SYSTEM_PROMPT


def augment_query(query:str) -> list[str]:
    if not query:
        print("Please enter a question.")
        return ""
    response = gemini_llm(query, system_prompt=AUGMENTATION_SYSTEM_PROMPT)
    if not response:
        return [query]
    queries = [line.strip() for line in response.split('\n') if line.strip()]
    if len(queries) < 3:
        return [query]
    return queries
