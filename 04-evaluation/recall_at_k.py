"""
Goal:
Show how Recall@K checks whether the correct chunk
appears in the Top K retrieved results.
"""

retrieved_chunks = [
    "AWS Glue throttling occurs when API request limits are exceeded.",
    "Glue job performance can improve by increasing worker capacity.",
    "AWS Glue jobs can retry failed requests automatically.",
    "Reduce concurrent API requests and use exponential backoff.",
    "Glue jobs can fail because of insufficient worker resources.",
]

correct_chunk = (
    "Reduce concurrent API requests and use exponential backoff."
)


def recall_at_k(retrieved, relevant, k):
    # Only whether the correct chunk made the cut matters, not its exact
    # rank within the top K -- that's what separates Recall@K from MRR/NDCG.
    top_k = retrieved[:k]
    return 1 if relevant in top_k else 0


# correct_chunk sits at index 3, so it's excluded at K=1 and K=3 but
# included at K=5 -- showing how the metric shifts as K grows.
for k in [1, 3, 5]:
    score = recall_at_k(
        retrieved_chunks,
        correct_chunk,
        k
    )

    print(f"Recall@{k}: {score}")