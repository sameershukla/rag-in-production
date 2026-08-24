"""
Chunking example for RAG.

Goal:
Show how chunk size can preserve or break useful context.
"""

# Sample source document that would normally be split into chunks before
# being embedded and stored in a vector index. It contains one coherent
# idea (a Glue throttling error) that spans multiple sentences.
DOCUMENT = """
AWS Glue jobs can fail with a ThrottlingException when too many API
requests are sent in a short period.

A common fix is to reduce concurrent requests and add exponential backoff.

The retry delay should gradually increase between failed attempts.
"""


def chunk_text(text: str, chunk_size: int) -> list[str]:
    # Split on whitespace into a flat list of words, ignoring line breaks.
    words = text.split()

    # Walk through the words in non-overlapping windows of chunk_size,
    # joining each window back into a chunk string. The last chunk may be
    # shorter than chunk_size if the word count doesn't divide evenly.
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


def show_chunks(label: str, chunks: list[str]) -> None:
    print(f"\n{label}")

    # Print each chunk with a 1-based index so it's easy to see how many
    # chunks were produced and how the text was divided between them.
    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}: {chunk}")


# Small chunk size (8 words): chunks are short enough that a single idea
# (e.g. the cause of the error vs. the fix vs. the retry detail) can end
# up split across multiple chunks, losing context if retrieved alone.
small_chunks = chunk_text(DOCUMENT, chunk_size=8)

# Large chunk size (25 words): each chunk holds enough words to keep
# related sentences together, preserving the full context of an idea.
large_chunks = chunk_text(DOCUMENT, chunk_size=25)

show_chunks("SMALL CHUNKS", small_chunks)
show_chunks("LARGE CHUNKS", large_chunks)