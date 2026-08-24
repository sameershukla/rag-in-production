"""
Embedding similarity example for RAG.

Goal:
Show how embeddings capture semantic similarity even when
two sentences do not use exactly the same words.
"""

# SentenceTransformer converts text into dense vector embeddings.
from sentence_transformers import SentenceTransformer
# cosine_similarity measures how close two vectors point in the same direction (1.0 = identical, 0 = unrelated).
from sklearn.metrics.pairwise import cosine_similarity


# Load a small, fast pretrained embedding model (384-dimensional vectors).
model = SentenceTransformer("all-MiniLM-L6-v2")

# The user's question we want to find relevant documents for.
query = "How do I reduce AWS Glue throttling?"

# Candidate documents to compare against the query. Note none of these
# repeat the exact words "reduce" + "AWS Glue" + "throttling" together,
# so a keyword search would struggle here.
documents = [
    "Reduce concurrent requests and use exponential backoff.",
    "AWS Glue can run Spark based ETL jobs.",
    "Amazon S3 provides object storage."
]


# Encode the query and documents into embedding vectors.
# encode() expects a list, so the query is wrapped in [query].
query_embedding = model.encode([query])
document_embeddings = model.encode(documents)

# Compute cosine similarity between the query vector and every document
# vector. Result is a 2D array (1 query x N documents), so [0] flattens
# it to a 1D array of scores, one per document.
scores = cosine_similarity(
    query_embedding,
    document_embeddings
)[0]


# Pair each document with its similarity score, then sort so the most
# semantically similar document (highest score) appears first.
for document, score in sorted(
    zip(documents, scores),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{score:.3f}  {document}")