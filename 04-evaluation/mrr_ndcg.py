"""
Goal:
Show how MRR and NDCG measure ranking quality.

MRR rewards finding the first relevant result early.
NDCG rewards placing highly relevant results near the top.
"""

import math


# 1 = relevant, 0 = not relevant
binary_relevance = [0, 1, 0, 0]

# Graded relevance:
# 2 = highly relevant
# 1 = somewhat relevant
# 0 = irrelevant
graded_relevance = [1, 2, 0, 0]


def reciprocal_rank(relevance):
    # Only the position of the FIRST relevant result matters here, so we
    # return as soon as we find it: rank 1 -> 1.0, rank 2 -> 0.5, etc.
    for rank, score in enumerate(relevance, start=1):
        if score > 0:
            return 1 / rank

    return 0.0


def dcg(relevance):
    # log2(rank + 1) grows slowly, so relevance found deep in the ranking
    # is discounted rather than dropped, but top ranks still dominate.
    return sum(
        score / math.log2(rank + 1)
        for rank, score in enumerate(relevance, start=1)
    )


def ndcg(relevance):
    # DCG alone isn't comparable across queries because it depends on how
    # much relevant material exists. Dividing by the DCG of the best
    # possible ordering (ideal) normalizes to a 0-1 scale where 1.0 means
    # the actual ranking was already optimal.
    ideal = sorted(relevance, reverse=True)

    ideal_dcg = dcg(ideal)

    if ideal_dcg == 0:
        return 0.0

    return dcg(relevance) / ideal_dcg


mrr = reciprocal_rank(binary_relevance)
ndcg_score = ndcg(graded_relevance)

print(f"MRR:  {mrr:.3f}")
print(f"NDCG: {ndcg_score:.3f}")

print(
    "\nLesson: MRR measures how quickly the first relevant result appears. "
    "NDCG measures how well the full ranking orders results by relevance."
)