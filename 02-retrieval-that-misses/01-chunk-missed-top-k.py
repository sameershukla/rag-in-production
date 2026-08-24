"""
Failure: The correct chunk exists but misses Top K.

Goal:
Show that relevant evidence can be present in the index
but still not reach the LLM.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Small, fast bi-encoder used to embed both the query and the chunks.
model = SentenceTransformer("all-MiniLM-L6-v2")

query = "How do I fix Glue job throttling?"

# Simulated chunk index — a mix of related and off-target passages.
chunks = [
    "AWS Glue throttling can occur when API request limits are exceeded.",
    "Glue job performance can improve by increasing worker capacity.",
    "AWS Glue retries failed API requests automatically.",
    "Reduce concurrent API requests and use exponential backoff.",
    "Glue jobs can fail because of insufficient worker resources.",
]

# The chunk that actually answers the query, tracked for the demo only.
correct_chunk = "Reduce concurrent API requests and use exponential backoff."

# Embed every chunk once, and the query once, into the same vector space.
embeddings = model.encode(chunks)
query_embedding = model.encode([query])

# Cosine similarity between the query vector and each chunk vector.
scores = cosine_similarity(query_embedding, embeddings)[0]

# Sort chunks by similarity score, highest first.
ranked = sorted(
    zip(chunks, scores),
    key=lambda x: x[1],
    reverse=True
)

# Only the top 3 ranked chunks would actually be passed to the LLM.
TOP_K = 3

print(f"Query: {query}\n")

for rank, (chunk, score) in enumerate(ranked, start=1):
    # Flag whether this chunk would make it past the Top K cutoff.
    retrieved = " <-- retrieved" if rank <= TOP_K else ""
    # Flag whether this is the chunk that actually answers the query.
    correct = " <-- correct answer" if chunk == correct_chunk else ""

    print(f"{rank}. {score:.3f} | {chunk}{retrieved}{correct}")

print(
    "\nLesson: The correct chunk exists, "
    "but Top K prevents it from reaching the LLM."
)