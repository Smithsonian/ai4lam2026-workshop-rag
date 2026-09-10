"""Exercise starter code for the data parsing tutorial module."""

import os


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split a long text string into smaller segments of a specified size with a given overlap.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum length of each resulting chunk in characters.
        overlap (int): The number of overlapping characters between consecutive chunks.

    Returns:
        list[str]: A list containing the segmented text strings.
    """
    # YOUR CODE HERE
    pass


if __name__ == "__main__":
    # Path to astronomy data
    data_dir = "dataset/astronomy_docs"
    files = [f for f in os.listdir(data_dir) if f.endswith(".md")]

    for file in files:
        with open(os.path.join(data_dir, file), "r") as f:
            content = f.read()
            chunks = chunk_text(content, chunk_size=100, overlap=20)
            print(f"File: {file} | Chunks generated: {len(chunks)}")
            for i, c in enumerate(chunks):
                print(f" Chunk {i}: {c[:50]}...")
