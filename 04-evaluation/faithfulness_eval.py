"""
Goal:
Show how faithfulness checks whether claims in the generated
answer are supported by the retrieved context.
"""

retrieved_context = """
AWS Glue throttling happens when too many API requests are sent.
To reduce throttling, reduce concurrent requests and use exponential backoff.
"""

answer_claims = [
    "AWS Glue throttling happens when too many API requests are sent.",
    "Reduce concurrent requests to reduce throttling.",
    "Use exponential backoff for retries.",
    "Increase worker capacity to eliminate throttling.",
]

# In practice this labeling is done by an LLM judge (or human) comparing
# each claim against retrieved_context. Hardcoded here since the point is
# the metric, not the judging step.
supported_claims = [
    True,
    True,
    True,
    False,  # not stated anywhere in retrieved_context -> a hallucination
]


# Faithfulness is the fraction of claims traceable to the retrieved context,
# not the fraction that happen to be factually true. A claim can be true in
# the real world and still be unfaithful if the context never said it.
faithfulness = (
    sum(supported_claims) / len(supported_claims)
)

print("Retrieved context:")
print(retrieved_context)

print("Answer claims:\n")

for claim, supported in zip(answer_claims, supported_claims):
    status = "SUPPORTED" if supported else "UNSUPPORTED"
    print(f"{status:11} | {claim}")

print(f"\nFaithfulness score: {faithfulness:.2f}")

print(
    "\nLesson: Faithfulness measures whether the generated answer "
    "is supported by the retrieved evidence."
)