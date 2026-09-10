"""Reference implementations for the data parsing tutorial module."""

import os


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into fixed-size chunks with a fixed character overlap."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # Move the pointer forward by (chunk_size - overlap)
        start += chunk_size - overlap
        if start >= len(text) or (end >= len(text)):
            break
    return chunks


def better_chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Chunk text while preferring word boundaries at chunk edges."""
    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        # 1. Clear leading whitespace
        while start < text_len and text[start] in [" ", "\n", "\t", "\r"]:
            start += 1

        if start >= text_len:
            break

        end = min(start + chunk_size, text_len)

        # 2. Snap `end` backward to a word boundary (unless at EOF)
        if end < text_len:
            boundary_end = end
            while boundary_end > start and text[boundary_end - 1] not in [" ", "\n"]:
                boundary_end -= 1
            if boundary_end > start:
                end = boundary_end

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        # 3. Target overlap start position
        target_start = end - overlap

        # 4. Snap `target_start` BACKWARD to a word boundary
        while target_start > 0 and text[target_start - 1] not in [" ", "\n"]:
            target_start -= 1

        # 5. Safeguard: Guarantee strictly positive progress.
        # If backward snapping pushes us to or before current `start`,
        # force advancement past `start` to avoid an infinite loop.
        if target_start <= start:
            next_start = start + 1
            # Advance to the start of the next clean word
            while next_start < end and text[next_start - 1] not in [" ", "\n"]:
                next_start += 1
            start = next_start
        else:
            start = target_start

    return chunks


if __name__ == "__main__":
    data_dir = "dataset/astronomy_docs"
    files = [f for f in os.listdir(data_dir) if f.endswith(".md")]

    for file in files:
        with open(os.path.join(data_dir, file), "r") as f:
            content = f.read()
            print(f"--- Processing {file} ---")
            N = 300
            print(f"\n[Character Chunking]: First N={N} chunks")
            chunks = chunk_text(content, chunk_size=150, overlap=20)
            for i, c in enumerate(chunks[:N]):
                print(f" Chunk {i}: {c}")
            print(f"\n[Improved Fixed-size Chunking]: First N={N} chunks")
            chunks = better_chunk_text(content, chunk_size=150, overlap=20)
            for i, c in enumerate(chunks[:N]):
                print(f" Chunk {i}: {c}")
            print("\n")
