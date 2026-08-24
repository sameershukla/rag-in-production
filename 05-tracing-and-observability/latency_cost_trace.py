"""
Goal:
Show how to trace latency and estimated cost
for different stages of a RAG request.
"""

import time


trace = {}


# Retrieval
# perf_counter (not time.time) is monotonic and immune to system clock
# adjustments, which matters for measuring short-lived spans accurately.
start = time.perf_counter()

time.sleep(0.08)   # simulate vector search

trace["retrieval_ms"] = (
    time.perf_counter() - start
) * 1000


# Generation
start = time.perf_counter()

time.sleep(0.35)   # simulate LLM generation

trace["generation_ms"] = (
    time.perf_counter() - start
) * 1000


# Token usage
input_tokens = 420
output_tokens = 32

# Output tokens are priced higher than input tokens (typical for LLM APIs),
# so a short answer over a huge context can still cost more per-token than
# it looks -- tracking the two separately surfaces that.
INPUT_COST_PER_1K = 0.003
OUTPUT_COST_PER_1K = 0.015

estimated_cost = (
    input_tokens / 1000 * INPUT_COST_PER_1K
    + output_tokens / 1000 * OUTPUT_COST_PER_1K
)


# Summed rather than measured end-to-end so total_ms stays consistent with
# the per-stage breakdown even if stages run sequentially with gaps
# between them (e.g. logging, post-processing) that aren't being traced.
trace["total_ms"] = (
    trace["retrieval_ms"]
    + trace["generation_ms"]
)

trace["estimated_cost_usd"] = estimated_cost


print("RAG Trace\n")

print(
    f"Retrieval latency : "
    f"{trace['retrieval_ms']:.1f} ms"
)

print(
    f"Generation latency: "
    f"{trace['generation_ms']:.1f} ms"
)

print(
    f"Total latency     : "
    f"{trace['total_ms']:.1f} ms"
)

print(
    f"Estimated cost    : "
    f"${trace['estimated_cost_usd']:.4f}"
)

print(
    "\nLesson: Trace each RAG stage separately so you can identify "
    "where latency and cost are coming from."
)