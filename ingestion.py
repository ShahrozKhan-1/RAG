# from pathlib import Path
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import re
# import pymupdf4llm
# import chromadb
# from config import CHROMA_COLLECTION, CHROMA_DIR, embedding

# MAX_TOKENS = 400
# PREFIX = "search_document: "
# TABLE_CONTINUED = "[Table Continued]"


# def count_tokens(text: str) -> int:
#     return len(text) // 4


# def split_table(lines: list[str], budget: int) -> list[str]:
#     # Find the separator line dynamically
#     sep_idx = next((i for i, line in enumerate(lines) if re.match(r"^\s*\|[-:\s|]+\|\s*$", line)), None)
#     if sep_idx is None:
#         raise ValueError("Markdown table missing separator row")

#     header = "\n".join(lines[: sep_idx + 1])
#     rows = lines[sep_idx + 1 :]
#     parts, buf = [], []
#     used = count_tokens(header)

#     for row in rows:
#         cost = count_tokens(row)
#         if buf and used + cost > budget:
#             continued = f"{TABLE_CONTINUED}\n" if parts else ""
#             parts.append(continued + header + "\n" + "\n".join(buf))
#             buf, used = [], count_tokens(header) + count_tokens(TABLE_CONTINUED)
#         buf.append(row)
#         used += cost

#     if buf:
#         continued = f"{TABLE_CONTINUED}\n" if parts else ""
#         parts.append(continued + header + "\n" + "\n".join(buf))
#     return parts


# def split_text(text: str, budget: int) -> list[str]:
#     if count_tokens(text) <= budget:
#         return [text]

#     lines = text.splitlines()
#     tables = []
#     i = 0
    
#     # Find all tables in the text
#     while i < len(lines):
#         if not lines[i].strip().startswith("|"):
#             i += 1
#             continue

#         sep_idx = i
#         while sep_idx < len(lines) and lines[sep_idx].strip().startswith("|") and not re.match(r"^\s*\|[-:\s|]+\|\s*$", lines[sep_idx]):
#             sep_idx += 1
            
#         if sep_idx >= len(lines) or not re.match(r"^\s*\|[-:\s|]+\|\s*$", lines[sep_idx]):
#             i += 1
#             continue

#         j = sep_idx + 1
#         while j < len(lines) and lines[j].strip().startswith("|"):
#             j += 1
#         tables.append((i, j))
#         i = j

#     if not tables:
#         return _split_paragraphs(text, budget)

#     parts, pos = [], 0
#     for start, end in tables:
#         before = "\n".join(lines[pos:start]).strip()
#         table_title = ""
        
#         # Simplified Title Detection (Catches # or Table/Figure)
#         if before:
#             preceding = before.splitlines()
#             last_line = preceding[-1].strip()
#             if last_line.startswith("#") or re.search(r"(?i)\b(table|figure)\b", last_line):
#                 table_title = preceding.pop()
#                 before = "\n".join(preceding).strip()

#         if before:
#             parts += _split_paragraphs(before, budget)

#         table_lines = lines[start:end]
#         if count_tokens("\n".join(table_lines)) <= budget:
#             parts.append(f"{table_title}\n" + "\n".join(table_lines) if table_title else "\n".join(table_lines))
#         else:
#             table_parts = split_table(table_lines, budget)
#             if table_title:
#                 table_parts = [
#                     f"{TABLE_CONTINUED}\n{table_title}\n{p[len(TABLE_CONTINUED):].lstrip()}" if p.startswith(TABLE_CONTINUED) else f"{table_title}\n{p}"
#                     for p in table_parts
#                 ]
#             parts += table_parts

#         pos = end

#     after = "\n".join(lines[pos:]).strip()
#     if after:
#         parts += _split_paragraphs(after, budget)

#     return parts


# def _split_paragraphs(text: str, budget: int) -> list[str]:
#     out, buf, used = [], [], 0
#     for para in text.split("\n\n"):
#         para = para.strip()
#         if not para:
#             continue
#         cost = count_tokens(para)
#         if cost > budget:
#             if buf:
#                 out.append("\n\n".join(buf))
#                 buf, used = [], 0
#             max_chars = budget * 4
#             while len(para) > max_chars:
#                 split_at = max_chars
#                 while split_at > 0 and not para[split_at].isspace():
#                     split_at -= 1
#                 if split_at == 0:
#                     split_at = max_chars
#                 out.append(para[:split_at].rstrip())
#                 para = para[split_at:].lstrip()
#             if para:
#                 out.append(para)
#             continue
#         if buf and used + cost > budget:
#             out.append("\n\n".join(buf))
#             buf, used = [], 0
#         buf.append(para)
#         used += cost
#     if buf:
#         out.append("\n\n".join(buf))
#     return out


# def merge_headings(chunks: list[dict]) -> list[dict]:
#     """Merges small isolated headings (#) with the following table."""
#     merged, i = [], 0
#     while i < len(chunks):
#         text = chunks[i]["text"].strip()
#         # If short heading and next chunk is a table -> merge
#         if i + 1 < len(chunks) and len(text) < 150 and text.startswith("#") and "|" in chunks[i+1]["text"] and "---" in chunks[i+1]["text"]:
#             chunks[i+1]["text"] = text + "\n\n" + chunks[i+1]["text"]
#             i += 1
#             continue
#         merged.append(chunks[i])
#         i += 1
#     return merged


# def split_oversized_chunks(chunks: list[dict]) -> list[dict]:
#     budget = MAX_TOKENS - count_tokens(PREFIX)
#     result = []
#     for chunk in chunks:
#         text = chunk["text"]
#         if count_tokens(text) <= budget:
#             result.append(chunk)
#             continue

#         pieces = split_text(text, budget)
#         print(f"  chunk {chunk['id']}: {count_tokens(text)} -> {len(pieces)} parts")

#         for i, piece in enumerate(pieces):
#             new = dict(chunk)
#             new["id"] = f"{chunk['id']}#{i}"
#             new["text"] = piece
#             new["metadata"] = {**chunk["metadata"], "token_count": count_tokens(piece)}
#             result.append(new)
#     return result


# def chunk_pdf(pdf_path: str, max_tokens: int = 350) -> list[dict]:
#     path = Path(pdf_path)
#     if not path.is_file():
#         raise FileNotFoundError(f"PDF not found: {path}")

#     document = pymupdf4llm.to_chunks(
#         str(path),
#         max_tokens=max_tokens,
#         table_mode="preserve",
#         respect_section_starts=True,
#         header_footer_mode="exclude",
#         write_images=False,
#         embed_images=False,
#     )
#     return [
#         {
#             "id": c.id,
#             "text": c.text,
#             "metadata": {
#                 "page_start": c.metadata.page_start,
#                 "page_end": c.metadata.page_end,
#                 "section_path": c.metadata.section_path,
#                 "types": c.metadata.types,
#                 "token_count": c.metadata.token_count,
#                 "table_ids": c.metadata.table_ids,
#                 "file_path": str(path),
#             },
#         }
#         for c in document
#     ]


# def embed_chunks(chunks: list[dict], max_workers: int = 10) -> list[dict]:
#     pending_limit = max_workers * 2
#     completed = 0
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         for start in range(0, len(chunks), pending_limit):
#             batch = chunks[start : start + pending_limit]
#             futures = {executor.submit(embedding, PREFIX + c["text"]): c for c in batch}
#             for future in as_completed(futures):
#                 chunk = futures[future]
#                 chunk["embedding"] = future.result()
#                 completed += 1
#                 print(f"Processed {completed}/{len(chunks)} chunks")
#     return chunks


# def save_to_chroma(chunks: list[dict], path: str | Path = CHROMA_DIR, batch_size: int = 500) -> None:
#     client = chromadb.PersistentClient(path=str(path))
#     collection = client.get_or_create_collection(name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
#     batch_size = min(batch_size, client.get_max_batch_size())

#     def clean(meta: dict) -> dict:
#         return {k: v for k, v in meta.items() if v not in (None, []) and isinstance(v, (str, int, float, bool, list))}

#     for start in range(0, len(chunks), batch_size):
#         batch = chunks[start : start + batch_size]
#         collection.upsert(
#             ids=[c["id"] for c in batch],
#             documents=[c["text"] for c in batch],
#             embeddings=[c["embedding"] for c in batch],
#             metadatas=[clean(c["metadata"]) for c in batch],
#         )
#         print(f"Saved {min(start + len(batch), len(chunks))}/{len(chunks)} chunks")

#     print(f"Saved {len(chunks)} chunks to {Path(path).resolve()}. Total in DB: {collection.count()}")


# # --- main ------------------------------------------------------------------
# if __name__ == "__main__":
#     pdf_path = input("Enter PDF path: ").strip().strip('"')

#     print("Extracting PDF...")
#     chunks = chunk_pdf(pdf_path)
#     print(f"Created {len(chunks)} chunks.")

#     print("Merging isolated headings with tables...")
#     chunks = merge_headings(chunks)

#     print(f"Splitting chunks over {MAX_TOKENS} tokens...")
#     chunks = split_oversized_chunks(chunks)
#     print(f"Now {len(chunks)} chunks.")

#     print("Generating embeddings...")
#     chunks = embed_chunks(chunks)

#     print("Saving to ChromaDB...")
#     save_to_chroma(chunks)



from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import pymupdf4llm
import chromadb
from config import CHROMA_COLLECTION, CHROMA_DIR, embedding

MAX_TOKENS = 400
PREFIX = "search_document: "
TABLE_CONTINUED = "[Table Continued]"


def count_tokens(text: str) -> int:
    return len(text) // 4


def _split_html_table(table_html: str, budget: int) -> list[str]:
    """Splits a massive HTML table by rows while preserving headers."""
    # Extract opening table tag to preserve attributes (border, class, etc.)
    table_open_match = re.match(r"<table.*?>", table_html, re.IGNORECASE | re.DOTALL)
    table_open = table_open_match.group(0) if table_open_match else "<table>"
    table_close = "</table>"
    
    # Extract thead
    thead_match = re.search(r"<thead.*?>.*?</thead>", table_html, re.IGNORECASE | re.DOTALL)
    
    # Find all rows
    rows = re.findall(r"<tr.*?>.*?</tr>", table_html, re.IGNORECASE | re.DOTALL)
    if not rows:
        return [table_html] # Fallback if malformed table
        
    header_html = ""
    body_rows = []
    
    if thead_match:
        header_html = thead_match.group(0)
        # Remove thead to find body rows
        body_html = table_html.replace(header_html, "")
        body_rows = re.findall(r"<tr.*?>.*?</tr>", body_html, re.IGNORECASE | re.DOTALL)
    else:
        # Assume first row is the header if no thead exists
        header_html = f"<thead>{rows[0]}</thead>"
        body_rows = rows[1:]
        
    if not body_rows:
        body_rows = rows
        header_html = ""

    parts, buf, used = [], [], count_tokens(table_open + header_html + table_close)
    
    for row in body_rows:
        cost = count_tokens(row)
        if buf and used + cost > budget:
            content = f"{TABLE_CONTINUED}\n{table_open}\n{header_html}\n<tbody>\n" + "\n".join(buf) + f"\n</tbody>\n{table_close}" if parts else f"{table_open}\n{header_html}\n<tbody>\n" + "\n".join(buf) + f"\n</tbody>\n{table_close}"
            parts.append(content)
            buf, used = [], count_tokens(table_open + header_html + table_close + TABLE_CONTINUED)
        buf.append(row)
        used += cost
        
    if buf:
        content = f"{TABLE_CONTINUED}\n{table_open}\n{header_html}\n<tbody>\n" + "\n".join(buf) + f"\n</tbody>\n{table_close}" if parts else f"{table_open}\n{header_html}\n<tbody>\n" + "\n".join(buf) + f"\n</tbody>\n{table_close}"
        parts.append(content)
        
    return parts


def _split_html_tables(text: str, budget: int) -> list[str]:
    """Finds HTML tables in text and splits text around them."""
    table_pattern = re.compile(r"<table.*?>.*?</table>", re.DOTALL | re.IGNORECASE)
    parts, pos = [], 0
    
    for match in table_pattern.finditer(text):
        before = text[pos:match.start()].strip()
        table_html = match.group(0)
        table_title = ""
        
        # Simplified Title Detection
        if before:
            preceding = before.splitlines()
            last_line = preceding[-1].strip()
            if last_line.startswith("#") or re.search(r"^<h\d>", last_line, re.IGNORECASE) or re.search(r"(?i)\b(table|figure)\b", last_line):
                table_title = preceding.pop()
                before = "\n".join(preceding).strip()

        if before:
            parts += _split_paragraphs(before, budget)

        if count_tokens(table_html) <= budget:
            parts.append(f"{table_title}\n{table_html}" if table_title else table_html)
        else:
            table_parts = _split_html_table(table_html, budget)
            if table_title:
                table_parts = [
                    f"{TABLE_CONTINUED}\n{table_title}\n{p[len(TABLE_CONTINUED):].lstrip()}" if p.startswith(TABLE_CONTINUED) else f"{table_title}\n{p}"
                    for p in table_parts
                ]
            parts += table_parts
            
        pos = match.end()
        
    after = text[pos:].strip()
    if after:
        parts += _split_paragraphs(after, budget)
        
    return parts


def split_text(text: str, budget: int) -> list[str]:
    if count_tokens(text) <= budget:
        return [text]

    # Check for HTML tables first
    if re.search(r"<table", text, re.IGNORECASE):
        return _split_html_tables(text, budget)

    return _split_paragraphs(text, budget)


def _split_paragraphs(text: str, budget: int) -> list[str]:
    out, buf, used = [], [], 0
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        cost = count_tokens(para)
        if cost > budget:
            if buf:
                out.append("\n\n".join(buf))
                buf, used = [], 0
            max_chars = budget * 4
            while len(para) > max_chars:
                split_at = max_chars
                while split_at > 0 and not para[split_at].isspace():
                    split_at -= 1
                if split_at == 0:
                    split_at = max_chars
                out.append(para[:split_at].rstrip())
                para = para[split_at:].lstrip()
            if para:
                out.append(para)
            continue
        if buf and used + cost > budget:
            out.append("\n\n".join(buf))
            buf, used = [], 0
        buf.append(para)
        used += cost
    if buf:
        out.append("\n\n".join(buf))
    return out


def merge_headings(chunks: list[dict]) -> list[dict]:
    """Merges small isolated headings (#) with the following HTML table."""
    merged, i = [], 0
    while i < len(chunks):
        text = chunks[i]["text"].strip()
        # If short heading and next chunk is an HTML table -> merge
        if i + 1 < len(chunks) and len(text) < 150 and (text.startswith("#") or re.match(r"^<h\d>", text, re.IGNORECASE)) and "<table" in chunks[i+1]["text"].lower():
            chunks[i+1]["text"] = text + "\n\n" + chunks[i+1]["text"]
            i += 1
            continue
        merged.append(chunks[i])
        i += 1
    return merged


def split_oversized_chunks(chunks: list[dict]) -> list[dict]:
    budget = MAX_TOKENS - count_tokens(PREFIX)
    result = []
    for chunk in chunks:
        text = chunk["text"]
        if count_tokens(text) <= budget:
            result.append(chunk)
            continue

        pieces = split_text(text, budget)
        print(f"  chunk {chunk['id']}: {count_tokens(text)} -> {len(pieces)} parts")

        for i, piece in enumerate(pieces):
            new = dict(chunk)
            new["id"] = f"{chunk['id']}#{i}"
            new["text"] = piece
            new["metadata"] = {**chunk["metadata"], "token_count": count_tokens(piece)}
            result.append(new)
    return result


def chunk_pdf(pdf_path: str, max_tokens: int = 350) -> list[dict]:
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    document = pymupdf4llm.to_chunks(
        str(path),
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
            "id": c.id,
            "text": c.text,
            "metadata": {
                "page_start": c.metadata.page_start,
                "page_end": c.metadata.page_end,
                "section_path": c.metadata.section_path,
                "types": c.metadata.types,
                "token_count": c.metadata.token_count,
                "table_ids": c.metadata.table_ids,
                "file_path": str(path),
            },
        }
        for c in document
    ]


def embed_chunks(chunks: list[dict], max_workers: int = 10) -> list[dict]:
    pending_limit = max_workers * 2
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for start in range(0, len(chunks), pending_limit):
            batch = chunks[start : start + pending_limit]
            futures = {executor.submit(embedding, PREFIX + c["text"]): c for c in batch}
            for future in as_completed(futures):
                chunk = futures[future]
                chunk["embedding"] = future.result()
                completed += 1
                print(f"Processed {completed}/{len(chunks)} chunks")
    return chunks


def save_to_chroma(chunks: list[dict], path: str | Path = CHROMA_DIR, batch_size: int = 500) -> None:
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
    batch_size = min(batch_size, client.get_max_batch_size())

    def clean(meta: dict) -> dict:
        return {k: v for k, v in meta.items() if v not in (None, []) and isinstance(v, (str, int, float, bool, list))}

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            metadatas=[clean(c["metadata"]) for c in batch],
        )
        print(f"Saved {min(start + len(batch), len(chunks))}/{len(chunks)} chunks")

    print(f"Saved {len(chunks)} chunks to {Path(path).resolve()}. Total in DB: {collection.count()}")


# --- main ------------------------------------------------------------------
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