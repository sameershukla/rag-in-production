"""
RRF combines ranked lists without needing to normalize or directly compare BM25 and dense scores.
Goal:
Show how Reciprocal Rank Fusion combines dense and BM25 rankings.

RRF does not combine raw scores.
It combines the positions of documents in each ranked list.
"""
from collections import defaultdict


dense_ranking = [
    "Glue job failures can be retried using exponential backoff.",
    "Job GLUE_JOB_9472 failed after repeated API throttling errors.",
    "AWS Glue jobs can fail because of throttling or insufficient capacity.",
    "AWS Glue worker capacity affects Spark job performance.",
]

bm25_ranking = [
    "Job GLUE_JOB_9472 failed after repeated API throttling errors.",
    "Glue job failures can be retried using exponential backoff.",
    "AWS Glue jobs can fail because of throttling or insufficient capacity.",
    "AWS Glue worker capacity affects Spark job performance.",
]


def reciprocal_rank_fusion(rankings, k=60):
    # A document's contribution from each list is 1/(k+rank), so rank 1
    # scores much higher than rank 2, but the gap shrinks for lower ranks.
    # k=60 is the constant from the original RRF paper; it dampens the
    # score so no single retriever's rank-1 pick can dominate the fusion.
    scores = defaultdict(float)

    for ranking in rankings:
        for rank, document in enumerate(ranking, start=1):
            scores[document] += 1 / (k + rank)

    # Sum scores across all rankings, then rank by the fused total.
    return sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


fused = reciprocal_rank_fusion(
    [dense_ranking, bm25_ranking]
)

print("Dense ranking:")
for rank, document in enumerate(dense_ranking, start=1):
    print(f"{rank}. {document}")

print("\nBM25 ranking:")
for rank, document in enumerate(bm25_ranking, start=1):
    print(f"{rank}. {document}")

print("\nRRF ranking:")
for rank, (document, score) in enumerate(fused, start=1):
    print(f"{rank}. {score:.4f} | {document}")

print(
    "\nLesson: RRF combines rank positions, not raw scores. "
    "Documents that consistently rank high across retrievers rise to the top."
)