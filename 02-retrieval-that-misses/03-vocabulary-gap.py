"""
Failure: Vocabulary gap between query and relevant chunk.

Goal:
Show that dense retrieval can rank a semantically related chunk
above the chunk containing the exact identifier.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Small, fast bi-encoder used to embed both the query and the chunks.
model = SentenceTransformer("all-MiniLM-L6-v2")

# Query centers on a specific job identifier rather than general wording.
query = "What happened to GLUE_JOB_9472?"

# Only one chunk mentions the exact job ID; the rest are generic Glue text.
chunks = [
    "AWS Glue jobs can fail because of throttling or insufficient capacity.",
    "Job GLUE_JOB_9472 failed after repeated API throttling errors.",
    "Glue job failures can be retried using exponential backoff.",
    "AWS Glue worker capacity affects Spark job performance.",
]

# The chunk containing the exact identifier, tracked for the demo only.
correct_chunk = (
    "Job GLUE_JOB_9472 failed after repeated API throttling errors."
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
    # Flag the chunk that literally contains the identifier from the query.
    correct = " <-- exact match" if chunk == correct_chunk else ""
    print(f"{rank}. {score:.3f} | {chunk}{correct}")