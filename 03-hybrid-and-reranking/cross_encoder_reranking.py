"""
Goal:
Show how a cross encoder can rerank retrieved chunks
so the passage that directly answers the query moves higher.
"""

from sentence_transformers import CrossEncoder


query = "How should I fix an AWS Glue TooManyRequestsException?"

candidates = [
    "TooManyRequestsException means AWS Glue is receiving too many API requests.",
    "To fix an AWS Glue TooManyRequestsException, reduce concurrent API requests and use exponential backoff.",
    "AWS Glue jobs use workers to process data with Apache Spark.",
    "AWS Glue supports automatic retries for some API operations.",
]

correct_chunk = (
    "To fix an AWS Glue TooManyRequestsException, "
    "reduce concurrent API requests and use exponential backoff."
)

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Unlike a bi-encoder (embed query and candidate separately, then compare
# vectors), a cross-encoder feeds the query and candidate through the model
# together as one input. That joint attention is what lets it tell a passage
# that directly answers the question apart from one merely on-topic.
pairs = [
    [query, candidate]
    for candidate in candidates
]

scores = model.predict(pairs)

ranked = sorted(
    zip(candidates, scores),
    key=lambda x: x[1],
    reverse=True
)

print(f"Query: {query}\n")

for rank, (candidate, score) in enumerate(ranked, start=1):
    correct = " <-- correct answer" if candidate == correct_chunk else ""
    print(f"{rank}. {score:.3f} | {candidate}{correct}")

print(
    "\nLesson: A cross encoder evaluates the query and each candidate "
    "together, helping distinguish a passage that directly answers "
    "the question from passages that are only related to the topic."
)