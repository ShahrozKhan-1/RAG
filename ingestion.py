from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re

import chromadb
import pymupdf4llm

from config import CHROMA_COLLECTION, CHROMA_DIR, embedding


MAX_TOKENS = 400
PREFIX = "search_document: "
TABLE_CONTINUED = "[Table Continued]"


def count_tokens(text: str) -> int:
    return len(text) // 4


def _format_html_table_chunk(
    table_open: str,
    header_html: str,
    rows: list[str],
    continued: bool,
) -> str:
    continuation_marker = f"{TABLE_CONTINUED}\n" if continued else ""
    row_html = "\n".join(rows)
    return (
        f"{continuation_marker}{table_open}\n"
        f"{header_html}\n"
        "<tbody>\n"
        f"{row_html}\n"
        "</tbody>\n</table>"
    )


def _split_html_table(table_html: str, budget: int) -> list[str]:
    """Split an HTML table into row groups while repeating its header."""
    opening_tag_match = re.match(
        r"<table.*?>",
        table_html,
        re.IGNORECASE | re.DOTALL,
    )
    opening_tag = opening_tag_match.group(0) if opening_tag_match else "<table>"

    thead_match = re.search(
        r"<thead.*?>.*?</thead>",
        table_html,
        re.IGNORECASE | re.DOTALL,
    )
    rows = re.findall(
        r"<tr.*?>.*?</tr>",
        table_html,
        re.IGNORECASE | re.DOTALL,
    )
    if not rows:
        return [table_html]

    if thead_match:
        header_html = thead_match.group(0)
        body_html = table_html.replace(header_html, "")
        body_rows = re.findall(
            r"<tr.*?>.*?</tr>",
            body_html,
            re.IGNORECASE | re.DOTALL,
        )
    else:
        header_html = f"<thead>{rows[0]}</thead>"
        body_rows = rows[1:]

    if not body_rows:
        body_rows = rows
        header_html = ""

    chunks: list[str] = []
    current_rows: list[str] = []
    current_token_count = count_tokens(
        opening_tag + header_html + "</table>"
    )

    for row in body_rows:
        row_token_count = count_tokens(row)
        if current_rows and current_token_count + row_token_count > budget:
            chunks.append(
                _format_html_table_chunk(
                    opening_tag,
                    header_html,
                    current_rows,
                    continued=bool(chunks),
                )
            )
            current_rows = []
            current_token_count = count_tokens(
                opening_tag + header_html + "</table>" + TABLE_CONTINUED
            )

        current_rows.append(row)
        current_token_count += row_token_count

    if current_rows:
        chunks.append(
            _format_html_table_chunk(
                opening_tag,
                header_html,
                current_rows,
                continued=bool(chunks),
            )
        )

    return chunks


def _split_html_tables(text: str, budget: int) -> list[str]:
    """Split text around HTML tables and preserve table titles."""
    table_pattern = re.compile(
        r"<table.*?>.*?</table>",
        re.DOTALL | re.IGNORECASE,
    )
    chunks: list[str] = []
    text_position = 0

    for table_match in table_pattern.finditer(text):
        preceding_text = text[text_position:table_match.start()].strip()
        table_html = table_match.group(0)
        table_title = ""

        if preceding_text:
            preceding_lines = preceding_text.splitlines()
            last_line = preceding_lines[-1].strip()
            is_heading = last_line.startswith("#") or re.search(
                r"^<h\d>",
                last_line,
                re.IGNORECASE,
            )
            names_table_or_figure = re.search(
                r"(?i)\b(table|figure)\b",
                last_line,
            )
            if is_heading or names_table_or_figure:
                table_title = preceding_lines.pop()
                preceding_text = "\n".join(preceding_lines).strip()

        if preceding_text:
            chunks.extend(_split_paragraphs(preceding_text, budget))

        if count_tokens(table_html) <= budget:
            chunks.append(
                f"{table_title}\n{table_html}" if table_title else table_html
            )
        else:
            table_chunks = _split_html_table(table_html, budget)
            if table_title:
                table_chunks = [
                    (
                        f"{TABLE_CONTINUED}\n{table_title}\n"
                        f"{table_chunk[len(TABLE_CONTINUED):].lstrip()}"
                        if table_chunk.startswith(TABLE_CONTINUED)
                        else f"{table_title}\n{table_chunk}"
                    )
                    for table_chunk in table_chunks
                ]
            chunks.extend(table_chunks)

        text_position = table_match.end()

    trailing_text = text[text_position:].strip()
    if trailing_text:
        chunks.extend(_split_paragraphs(trailing_text, budget))

    return chunks


def split_text(text: str, budget: int) -> list[str]:
    if count_tokens(text) <= budget:
        return [text]

    contains_html_table = re.search(r"<table", text, re.IGNORECASE)
    if contains_html_table:
        return _split_html_tables(text, budget)

    return _split_paragraphs(text, budget)


def _split_long_paragraph(paragraph: str, max_characters: int) -> list[str]:
    chunks: list[str] = []
    while len(paragraph) > max_characters:
        split_position = max_characters
        while split_position > 0 and not paragraph[split_position].isspace():
            split_position -= 1
        if split_position == 0:
            split_position = max_characters

        chunks.append(paragraph[:split_position].rstrip())
        paragraph = paragraph[split_position:].lstrip()

    if paragraph:
        chunks.append(paragraph)
    return chunks


def _split_paragraphs(text: str, budget: int) -> list[str]:
    chunks: list[str] = []
    current_paragraphs: list[str] = []
    current_token_count = 0

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        paragraph_token_count = count_tokens(paragraph)
        if paragraph_token_count > budget:
            if current_paragraphs:
                chunks.append("\n\n".join(current_paragraphs))
                current_paragraphs = []
                current_token_count = 0

            chunks.extend(_split_long_paragraph(paragraph, budget * 4))
            continue

        if (
            current_paragraphs
            and current_token_count + paragraph_token_count > budget
        ):
            chunks.append("\n\n".join(current_paragraphs))
            current_paragraphs = []
            current_token_count = 0

        current_paragraphs.append(paragraph)
        current_token_count += paragraph_token_count

    if current_paragraphs:
        chunks.append("\n\n".join(current_paragraphs))

    return chunks


def merge_headings(chunks: list[dict]) -> list[dict]:
    """Merge isolated headings into the following HTML table chunk."""
    merged_chunks: list[dict] = []
    chunk_index = 0

    while chunk_index < len(chunks):
        current_chunk = chunks[chunk_index]
        heading = current_chunk["text"].strip()
        has_following_chunk = chunk_index + 1 < len(chunks)

        if has_following_chunk:
            following_chunk = chunks[chunk_index + 1]
            is_short_heading = len(heading) < 150 and (
                heading.startswith("#")
                or re.match(r"^<h\d>", heading, re.IGNORECASE)
            )
            following_chunk_has_table = (
                "<table" in following_chunk["text"].lower()
            )
            if is_short_heading and following_chunk_has_table:
                following_chunk["text"] = (
                    heading + "\n\n" + following_chunk["text"]
                )
                chunk_index += 1
                continue

        merged_chunks.append(current_chunk)
        chunk_index += 1

    return merged_chunks


def split_oversized_chunks(chunks: list[dict]) -> list[dict]:
    token_budget = MAX_TOKENS - count_tokens(PREFIX)
    split_chunks: list[dict] = []

    for chunk in chunks:
        text = chunk["text"]
        if count_tokens(text) <= token_budget:
            split_chunks.append(chunk)
            continue

        text_parts = split_text(text, token_budget)
        for part_index, text_part in enumerate(text_parts):
            split_chunk = dict(chunk)
            split_chunk["id"] = f"{chunk['id']}#{part_index}"
            split_chunk["text"] = text_part
            split_chunk["metadata"] = {
                **chunk["metadata"],
                "token_count": count_tokens(text_part),
            }
            split_chunks.append(split_chunk)

    return split_chunks


def chunk_pdf(pdf_path: str, max_tokens: int = 350) -> list[dict]:
    pdf_file = Path(pdf_path)
    if not pdf_file.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    document_chunks = pymupdf4llm.to_chunks(
        str(pdf_file),
        max_tokens=max_tokens,
        table_output="html",
        table_mode="preserve",
        respect_section_starts=True,
        header_footer_mode="exclude",
        write_images=False,
        embed_images=False,
    )
    return [
        {
            "id": chunk.id,
            "text": chunk.text,
            "metadata": {
                "page_start": chunk.metadata.page_start,
                "page_end": chunk.metadata.page_end,
                "section_path": chunk.metadata.section_path,
                "types": chunk.metadata.types,
                "token_count": chunk.metadata.token_count,
                "table_ids": chunk.metadata.table_ids,
                "file_path": str(pdf_file),
            },
        }
        for chunk in document_chunks
    ]


def embed_chunks(chunks: list[dict], max_workers: int = 10) -> list[dict]:
    pending_limit = max_workers * 2
    completed_chunks = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch_start in range(0, len(chunks), pending_limit):
            batch = chunks[batch_start:batch_start + pending_limit]
            embedding_futures = {
                executor.submit(embedding, PREFIX + chunk["text"]): chunk
                for chunk in batch
            }
            for future in as_completed(embedding_futures):
                chunk = embedding_futures[future]
                chunk["embedding"] = future.result()
                completed_chunks += 1
                print(f"Processed {completed_chunks}/{len(chunks)} chunks")

    return chunks


def _clean_metadata(metadata: dict) -> dict:
    return {
        key: value
        for key, value in metadata.items()
        if value not in (None, [])
        and isinstance(value, (str, int, float, bool, list))
    }


def save_to_chroma(
    chunks: list[dict],
    path: str | Path = CHROMA_DIR,
    batch_size: int = 500,
) -> None:
    chroma_client = chromadb.PersistentClient(path=str(path))
    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    batch_size = min(batch_size, chroma_client.get_max_batch_size())

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start:batch_start + batch_size]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["text"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=[
                _clean_metadata(chunk["metadata"])
                for chunk in batch
            ],
        )
        saved_count = min(batch_start + len(batch), len(chunks))
        print(f"Saved {saved_count}/{len(chunks)} chunks")

    print(
        f"Saved {len(chunks)} chunks to {Path(path).resolve()}. "
        f"Total in DB: {collection.count()}"
    )


if __name__ == "__main__":
    pdf_path = input("Enter PDF path: ").strip().strip('"')

    print("Extracting PDF...")
    chunks = chunk_pdf(pdf_path)
    print(f"Created {len(chunks)} chunks.")

    print("Merging isolated headings with tables...")
    chunks = merge_headings(chunks)

    print(f"Splitting chunks over {MAX_TOKENS} tokens...")
    chunks = split_oversized_chunks(chunks)
    print(f"Now {len(chunks)} chunks.")

    print("Generating embeddings...")
    chunks = embed_chunks(chunks)

    print("Saving to ChromaDB...")
    save_to_chroma(chunks)
