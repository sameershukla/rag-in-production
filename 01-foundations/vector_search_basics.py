"""
Basic vector search for RAG.

Goal:
Show how a query is matched against stored chunk embeddings
and how the Top K results are selected.
"""

# SentenceTransformer turns text chunks into dense vector embeddings.
from sentence_transformers import SentenceTransformer
# cosine_similarity scores how closely two vectors align (1.0 = identical, 0 = unrelated).
from sklearn.metrics.pairwise import cosine_similarity


# Load a small, fast pretrained embedding model (384-dimensional vectors).
model = SentenceTransformer("all-MiniLM-L6-v2")

# Simulated "vector store" contents: pre-chunked pieces of documents that
# would normally already be embedded and indexed ahead of time.
chunks = [
    "Reduce concurrent AWS Glue requests and use exponential backoff.",
    "AWS Glue jobs can process data using Apache Spark.",
    "Amazon S3 is an object storage service.",
    "Glue throttling can happen when too many API requests are sent."
]

# The incoming user question we need to answer using the chunks above.
query = "How can I fix AWS Glue throttling?"

# Embed all chunks (the "index") and the query using the same model, so
# they live in the same vector space and can be compared directly.
chunk_embeddings = model.encode(chunks)
query_embedding = model.encode([query])

# Compare the query vector against every chunk vector at once.
# Result is a 2D array (1 query x N chunks), so [0] flattens it to a
# 1D array of similarity scores, one per chunk.
scores = cosine_similarity(query_embedding, chunk_embeddings)[0]

# Number of top-matching chunks to retrieve, i.e. "Top K" retrieval.
top_k = 2

# Pair each chunk with its score, sort by score descending (most
# relevant first), then keep only the top_k highest-scoring chunks.
results = sorted(
    zip(chunks, scores),
    key=lambda item: item[1],
    reverse=True
)[:top_k]

print(f"Query: {query}\n")

# Print the retrieved chunks in rank order along with their similarity
# scores, mimicking what a retriever would hand off to the LLM.
for rank, (chunk, score) in enumerate(results, start=1):
    print(f"{rank}. {score:.3f}  {chunk}")