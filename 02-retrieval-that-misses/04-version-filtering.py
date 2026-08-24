"""
Failure: Retrieval returns the wrong document version.

Goal:
Show that semantic similarity alone can retrieve stale content
unless version metadata is used as a filter.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Small, fast bi-encoder used to embed both the query and the documents.
model = SentenceTransformer("all-MiniLM-L6-v2")

query = "How should Glue throttling be handled?"

# Two documents say almost the same thing, but only one reflects the
# current (2026) guidance — the other is stale advice from 2024.
documents = [
    {
        "text": "Reduce concurrent requests and use fixed 5-second retries.",
        "version": "2024",
    },
    {
        "text": "Reduce concurrent requests and use exponential backoff.",
        "version": "2026",
    },
    {
        "text": "Increase Glue worker capacity for Spark processing.",
        "version": "2026",
    },
]

# Pull out just the text for embedding; metadata stays attached via zip.
texts = [doc["text"] for doc in documents]

# Embed every document once, and the query once, into the same vector space.
embeddings = model.encode(texts)
query_embedding = model.encode([query])

# Cosine similarity between the query vector and each document vector.
scores = cosine_similarity(query_embedding, embeddings)[0]

# Sort documents by similarity score, highest first.
ranked = sorted(
    zip(documents, scores),
    key=lambda x: x[1],
    reverse=True,
)

print(f"Query: {query}\n")

# Similarity alone can't distinguish current guidance from outdated advice,
# so the stale 2024 doc can easily outrank the current one.
print("Without version filtering:")
for rank, (doc, score) in enumerate(ranked, start=1):
    print(
        f"{rank}. {score:.3f} | "
        f"{doc['text']} | version={doc['version']}"
    )


current_version = "2026"

# Keep only documents matching the current version, preserving their rank order.
filtered = [
    (doc, score)
    for doc, score in ranked
    if doc["version"] == current_version
]

print(f"\nWith version={current_version} filter:")
for rank, (doc, score) in enumerate(filtered, start=1):
    print(
        f"{rank}. {score:.3f} | "
        f"{doc['text']} | version={doc['version']}"
    )