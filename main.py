from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from augmentation import augment_query
from config import gemini_llm
from reranker import rerank_chunks
from retrieval import retrieve
from system_prompt import RETRIEVAL_SYSTEM_PROMPT


_RETRIEVAL_CANDIDATE_LIMIT = 15


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    sources: list[str]


class NoRelevantContextError(Exception):
    """Raised when retrieval did not find any documents for a query."""


def _format_source(index: int, hit: dict) -> str:
    metadata = hit.get("metadata") or {}
    source_name = Path(metadata.get("file_path") or "Retrieved document").name
    section_path = metadata.get("section_path") or []
    if isinstance(section_path, str):
        section = section_path
    else:
        section = " › ".join(str(part) for part in section_path)

    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")
    if page_start is None:
        page_label = ""
    elif page_end is not None and page_end != page_start:
        page_label = f", pages {page_start}–{page_end}"
    else:
        page_label = f", page {page_start}"

    section_label = f", {section}" if section else ""
    return f"[{index}] {source_name}{page_label}{section_label}"


def _build_answer_prompt(
    query: str,
    ranked_hits: list[dict],
    chat_history: list[dict[str, str]],
) -> str:
    context_sections = []
    for source_index, hit in enumerate(ranked_hits, start=1):
        source_description = _format_source(source_index, hit)
        context_sections.append(
            f"{source_description}\n"
            f"{hit.get('text') or ''}"
        )

    prompt_sections = [
        "Retrieved context:\n" + "\n\n".join(context_sections),
    ]
    if chat_history:
        conversation = "\n".join(
            f"{message['role'].title()}: {message['content']}"
            for message in chat_history
        )
        prompt_sections.append(f"Conversation so far:\n{conversation}")

    prompt_sections.append(f"Current user question:\n{query}")
    return "\n\n".join(prompt_sections)


def get_rag_result(
    user_query: str,
    chat_history: list[dict[str, str]] | None = None,
) -> RAGResponse:
    """Run query augmentation, retrieval, reranking, and answer generation."""
    normalized_query = user_query.strip()
    if not normalized_query:
        raise ValueError("Enter a question before submitting.")

    conversation_history = chat_history or []
    search_queries = augment_query(
        normalized_query,
        chat_history=conversation_history,
    )
    if isinstance(search_queries, str):
        search_queries = [search_queries] if search_queries else [normalized_query]

    retrieved_hits: list[dict] = []
    seen_chunk_ids: set[str] = set()
    retrieval_errors: list[Exception] = []

    with ThreadPoolExecutor(max_workers=len(search_queries)) as executor:
        retrieval_futures = [
            executor.submit(retrieve, search_query)
            for search_query in search_queries
        ]
        for future in as_completed(retrieval_futures):
            try:
                for hit in future.result():
                    chunk_id = hit["id"]
                    if chunk_id in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(chunk_id)
                    retrieved_hits.append(hit)
            except Exception as error:
                retrieval_errors.append(error)

    if not retrieved_hits:
        if retrieval_errors:
            raise RuntimeError(
                "The document search failed for every generated query."
            ) from retrieval_errors[0]
        raise NoRelevantContextError(
            "I couldn't find relevant passages in the indexed documents."
        )

    candidate_hits = retrieved_hits[:_RETRIEVAL_CANDIDATE_LIMIT]
    ranked_hits = rerank_chunks(
        query=normalized_query,
        chunks=candidate_hits,
    )
    if not ranked_hits:
        raise NoRelevantContextError(
            "I couldn't find relevant passages in the indexed documents."
        )

    prompt = _build_answer_prompt(
        normalized_query,
        ranked_hits,
        conversation_history,
    )
    answer = gemini_llm(
        query=prompt,
        system_prompt=RETRIEVAL_SYSTEM_PROMPT,
    )
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("The language model returned an empty response.")

    sources = [
        _format_source(source_index, hit)
        for source_index, hit in enumerate(ranked_hits, start=1)
    ]
    return RAGResponse(answer=answer, sources=sources)


def get_rag_response(user_query: str) -> str:
    """Return an answer without source metadata for simple callers."""
    return get_rag_result(user_query).answer


def main() -> None:
    query = input("Question: ").strip()
    if not query:
        print("Please enter a question.")
        return

    try:
        response = get_rag_result(query)
    except NoRelevantContextError:
        print("No results.")
        return
    except Exception as error:
        print(f"Unable to answer the question: {error}")
        return

    print(response.answer)


if __name__ == "__main__":
    main()
