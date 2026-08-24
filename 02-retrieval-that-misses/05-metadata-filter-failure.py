"""
Failure: Correct chunk is removed by a metadata filter.

Goal:
Show that bad or inconsistent metadata can hide relevant
content even when the chunk exists in the index.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Small, fast bi-encoder used to embed both the query and the documents.
model = SentenceTransformer("all-MiniLM-L6-v2")

query = "How do I fix AWS Glue throttling?"

# The best answer is tagged "AWS-Glue" while the other docs are tagged
# "Glue" — an inconsistent metadata value for what is really the same service.
documents = [
    {
        "text": "Reduce concurrent AWS Glue API requests and use exponential backoff.",
        "service": "AWS-Glue",
    },
    {
        "text": "AWS Glue throttling occurs when API request limits are exceeded.",
        "service": "Glue",
    },
    {
        "text": "Increase worker capacity to improve Spark job performance.",
        "service": "Glue",
    },
]

required_service = "Glue"

# Strict equality match on the metadata filter — "AWS-Glue" != "Glue",
# so the best-answer document is dropped before retrieval even scores it.
filtered_documents = [
    doc
    for doc in documents
    if doc["service"] == required_service
]

# Pull out just the text for embedding; metadata stays attached via zip.
texts = [doc["text"] for doc in filtered_documents]

# Embed every remaining document once, and the query once, into the same
# vector space.
embeddings = model.encode(texts)
query_embedding = model.encode([query])

# Cosine similarity between the query vector and each remaining document vector.
scores = cosine_similarity(query_embedding, embeddings)[0]

# Sort the surviving documents by similarity score, highest first.
ranked = sorted(
    zip(filtered_documents, scores),
    key=lambda x: x[1],
    reverse=True,
)

print(f"Query: {query}")
print(f"Filter: service={required_service}\n")

for rank, (doc, score) in enumerate(ranked, start=1):
    print(
        f"{rank}. {score:.3f} | "
        f"{doc['text']} | service={doc['service']}"
    )

# The actual best answer, from the original unfiltered document list.
correct_chunk = documents[0]

# Confirm the metadata filter silently discarded the correct chunk before
# it ever had a chance to be ranked.
if correct_chunk not in filtered_documents:
    print(
        "\nFAILURE: Correct chunk exists, "
        "but metadata filtering removed it."
    )