"""
Dense search is good at meaning. BM25 is good at exact words. Hybrid search combines both.
A direct fix of the issue we saw in 03-vocabulary-gap.py
Goal:
Show how hybrid retrieval combines dense semantic search
with BM25 keyword search.
"""
import re

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2")

query = "What happened to GLUE_JOB_9472?"

documents = [
    "Glue job failures can be retried using exponential backoff.",
    "Job GLUE_JOB_9472 failed after repeated API throttling errors.",
    "AWS Glue jobs can fail because of throttling or insufficient capacity.",
    "AWS Glue worker capacity affects Spark job performance.",
]


def tokenize(text):
    # Keep hyphenated identifiers like "GLUE_JOB_9472" intact instead of
    # splitting on underscores/hyphens, since BM25 matches on exact tokens.
    return re.findall(r"\b[\w-]+\b", text.lower())


# Dense search: embed query and documents into the same vector space, then
# rank by semantic similarity. This is what catches meaning-based matches
# even when the wording differs from the query.
doc_embeddings = model.encode(documents)
query_embedding = model.encode([query])

dense_scores = cosine_similarity(
    query_embedding,
    doc_embeddings
)[0]


# BM25 search: ranks by exact term overlap, so it's what actually finds the
# literal "GLUE_JOB_9472" identifier that dense search may blur together
# with semantically similar but non-matching documents.
tokenized_docs = [tokenize(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

bm25_scores = bm25.get_scores(
    tokenize(query)
)


# Normalize scores
def normalize(scores):
    # Cosine similarity and BM25 live on different, incomparable scales
    # (roughly 0-1 vs. unbounded). Min-max scaling puts both retrievers'
    # scores into a common 0-1 range so they can be blended fairly below.
    minimum = min(scores)
    maximum = max(scores)

    if maximum == minimum:
        return [0.0] * len(scores)

    return [
        (score - minimum) / (maximum - minimum)
        for score in scores
    ]


dense_normalized = normalize(dense_scores)
bm25_normalized = normalize(bm25_scores)

# Equal-weight blend of the two normalized scores. In practice this weight
# is tuned per corpus; 0.5/0.5 is a neutral starting point.
hybrid_scores = [
    0.5 * dense + 0.5 * keyword
    for dense, keyword in zip(
        dense_normalized,
        bm25_normalized
    )
]

# Rank by the blended hybrid score (index 3), not either retriever alone.
ranked = sorted(
    zip(documents, dense_scores, bm25_scores, hybrid_scores),
    key=lambda x: x[3],
    reverse=True
)

print(f"Query: {query}\n")

for rank, (doc, dense, bm25_score, hybrid) in enumerate(ranked, start=1):
    print(
        f"{rank}. "
        f"Hybrid={hybrid:.3f} | "
        f"Dense={dense:.3f} | "
        f"BM25={bm25_score:.3f} | "
        f"{doc}"
    )