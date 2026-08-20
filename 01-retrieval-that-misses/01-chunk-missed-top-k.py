# scenario_01_chunk_missed_top_k.py

"""
Scenario 1: The Chunk That Never Made It into Top K

Problem:
The correct chunk exists in the corpus, but dense retrieval
ranks it below top_k.

Fix:
Use dense + BM25 retrieval and combine the rankings using RRF.
"""

# SentenceTransformer is used for dense retrieval. It converts text into vectors, or embeddings.
from sentence_transformers import SentenceTransformer
# BM25Okapi is used for lexical retrieval. BM25 looks at the actual words appearing in the query and documents.
from rank_bm25 import BM25Okapi
# numpy is used mainly for vector similarity calculations and sorting the scores.
import numpy as np


# ---------------------------------------------------------
# Sample corpus
# ---------------------------------------------------------

documents = [
    {
        "chunk_id": "c1",
        "source": "glue_common_failures_general.md",
        "text": """
        AWS Glue ETL jobs commonly fail because of data quality
        issues, schema evolution, permissions, or resource problems.
        """
    },
    {
        "chunk_id": "c2",
        "source": "aws_glue_troubleshooting_guide.md",
        "text": """
        When troubleshooting AWS Glue jobs, review CloudWatch logs,
        IAM permissions, worker configuration, and source data.
        """
    },
    {
        "chunk_id": "c3",
        "source": "glue_cloudwatch_debugging.md",
        "text": """
        CloudWatch logs contain detailed information about failed
        AWS Glue jobs, including Spark exceptions and resource errors.
        """
    },
    {
        "chunk_id": "c4",
        "source": "glue_schema_problems.md",
        "text": """
        Schema mismatches can occur when source columns do not match
        the data types or columns stored in the Glue Catalog.
        """
    },
    {
        "chunk_id": "c5",
        "source": "glue_error_codes_internal.md",
        "text": """
        GLUE_ETL_ERROR_2027 indicates a schema mismatch between
        the source data and the target Glue Catalog table.

        Check the source dataframe schema against the target table
        columns and verify that the Glue crawler ran successfully.
        """
    },
]

# User query
query = (
    "What does GLUE_ETL_ERROR_2027 mean "
    "and how do I fix it?"
)

expected_source = "glue_error_codes_internal.md"

# This means the application will eventually send only three chunks downstream.
TOP_K = 3


# ---------------------------------------------------------
# Build dense index
# ---------------------------------------------------------

# Loading the embedding model. The job is converting the text into numeric vectors.
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [doc["text"] for doc in documents]
# Embed all documents. Why normalize_embeddings=True? This makes each embedding vector have length 1.
document_vectors = model.encode(
    texts,
    normalize_embeddings=True
)

# ---------------------------------------------------------
# Dense retrieval
# ---------------------------------------------------------

def dense_search(query, top_k=3):

    # Embedding of the query
    query_vector = model.encode(
        query,
        normalize_embeddings=True
    )

    # Calculate similarity scores of document vectors and query vector
    scores = np.dot(
        document_vectors,
        query_vector
    )

    # Sort Results by Score
    ranked_indexes = np.argsort(scores)[::-1]

    results = []

    for index in ranked_indexes[:top_k]:
        results.append({
            **documents[index],
            "score": float(scores[index])
        })

    # Returning the dense results
    return results


# Same idea used in Scenario 1:
# inspect a larger ranking to find where the expected chunk landed.
def find_expected_rank(query, expected_source):

    results = dense_search(
        query,
        top_k=len(documents)
    )

    for rank, chunk in enumerate(results, start=1):

        if chunk["source"] == expected_source:
            return rank

    return -1


# ---------------------------------------------------------
# Show the failure
# ---------------------------------------------------------

print("\n--- Dense Retrieval ---")

dense_results = dense_search(
    query,
    top_k=TOP_K
)

for rank, chunk in enumerate(
    dense_results,
    start=1
):
    print(
        rank,
        chunk["source"],
        round(chunk["score"], 3)
    )


rank = find_expected_rank(
    query,
    expected_source
)

print(
    "\nExpected source rank:",
    rank
)


found = any(
    chunk["source"] == expected_source
    for chunk in dense_results
)

print(
    "Correct chunk in Top K:",
    found
)


# ---------------------------------------------------------
# BM25
# ---------------------------------------------------------

tokenized_documents = [
    doc["text"].lower().split()
    for doc in documents
]

bm25_index = BM25Okapi(
    tokenized_documents
)


def bm25_search(query, top_k=3):

    query_tokens = query.lower().split()

    scores = bm25_index.get_scores(
        query_tokens
    )

    ranked_indexes = np.argsort(scores)[::-1]

    results = []

    for index in ranked_indexes[:top_k]:
        results.append({
            **documents[index],
            "score": float(scores[index])
        })

    return results


print("\n--- BM25 Retrieval ---")

bm25_results = bm25_search(
    query,
    top_k=TOP_K
)

for rank, chunk in enumerate(
    bm25_results,
    start=1
):
    print(
        rank,
        chunk["source"],
        round(chunk["score"], 3)
    )


# ---------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------

def reciprocal_rank_fusion(
    *result_lists,
    k=60
):

    scores = {}
    chunk_by_id = {}

    for results in result_lists:

        for rank, chunk in enumerate(
            results,
            start=1
        ):

            chunk_id = chunk["chunk_id"]

            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + 1.0 / (k + rank)
            )

            chunk_by_id[chunk_id] = chunk


    ranked_ids = sorted(
        scores,
        key=lambda chunk_id: scores[chunk_id],
        reverse=True
    )

    return [
        {
            **chunk_by_id[chunk_id],
            "rrf_score": scores[chunk_id]
        }
        for chunk_id in ranked_ids
    ]


# ---------------------------------------------------------
# Fixed retrieval
# ---------------------------------------------------------

def retrieve(query, top_k=3):

    # Retrieve a wider candidate pool first.
    dense_results = dense_search(
        query,
        top_k=5
    )

    bm25_results = bm25_search(
        query,
        top_k=5
    )

    # Fuse the two rankings.
    fused_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results
    )

    # Only now select final Top K.
    return fused_results[:top_k]


# ---------------------------------------------------------
# Verify the fix
# ---------------------------------------------------------

print("\n--- Hybrid Retrieval ---")

hybrid_results = retrieve(
    query,
    top_k=TOP_K
)

for rank, chunk in enumerate(
    hybrid_results,
    start=1
):

    print(
        rank,
        chunk["source"],
        round(chunk["rrf_score"], 4)
    )


found_after_fix = any(
    chunk["source"] == expected_source
    for chunk in hybrid_results
)

print(
    "\nCorrect chunk in Top K after fix:",
    found_after_fix
)
