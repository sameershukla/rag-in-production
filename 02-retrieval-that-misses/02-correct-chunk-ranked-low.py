"""
Failure: The correct chunk is retrieved, but ranked too low.

Goal:
Show that semantic similarity can rank a related chunk
above the chunk that actually contains the best answer.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Small, fast bi-encoder used to embed both the query and the chunks.
model = SentenceTransformer("all-MiniLM-L6-v2")

query = "How do I fix AWS Glue throttling?"

# Chunks are all topically related to Glue, but only one holds the fix.
chunks = [
    "AWS Glue throttling occurs when API request limits are exceeded.",
    "Reduce concurrent AWS Glue API requests and use exponential backoff.",
    "AWS Glue jobs can retry failed requests automatically.",
    "Increasing Glue worker capacity can improve job performance.",
]

# The chunk that actually answers the query, tracked for the demo only.
correct_chunk = (
    "Reduce concurrent AWS Glue API requests and use exponential backoff."
)

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

print(f"Query: {query}\n")

for rank, (chunk, score) in enumerate(ranked, start=1):
    # Flag whether this is the chunk that actually answers the query.
    correct = " <-- correct answer" if chunk == correct_chunk else ""

    print(f"{rank}. {score:.3f} | {chunk}{correct}")

# Find where the correct chunk actually landed in the ranking.
correct_rank = next(
    rank
    for rank, (chunk, _) in enumerate(ranked, start=1)
    if chunk == correct_chunk
)

print(f"\nCorrect chunk rank: {correct_rank}")

# Even though the correct chunk was retrieved, it lost the top spot to
# a chunk that only restates the problem instead of solving it.
if correct_rank > 1:
    print("FAILURE: Correct chunk was retrieved, but ranked below a weaker answer.")