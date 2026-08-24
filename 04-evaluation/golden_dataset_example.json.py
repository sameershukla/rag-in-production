# expected_chunk: the passage retrieval must surface, used to score
#   Recall@K / MRR / NDCG independently of what the LLM generates.
# expected_answer: the reference answer, used to score generation quality
#   (e.g. faithfulness) separately from retrieval quality.
# expected_metadata: present only when a case is testing metadata
#   filtering (e.g. rag-003 needs the 2026-version chunk specifically, not
#   just any chunk with the right text).
[
  {
    "id": "rag-001",
    "question": "How do I fix AWS Glue throttling?",
    "expected_chunk": "Reduce concurrent AWS Glue API requests and use exponential backoff.",
    "expected_answer": "Reduce concurrent API requests and use exponential backoff."
  },
  {
    "id": "rag-002",
    "question": "What happened to GLUE_JOB_9472?",
    "expected_chunk": "Job GLUE_JOB_9472 failed after repeated API throttling errors.",
    "expected_answer": "GLUE_JOB_9472 failed because of repeated API throttling errors."
  },
  {
    "id": "rag-003",
    "question": "Which guidance should be used for Glue throttling?",
    "expected_chunk": "Reduce concurrent requests and use exponential backoff.",
    "expected_metadata": {
      "version": "2026"
    },
    "expected_answer": "Use the 2026 guidance: reduce concurrent requests and use exponential backoff."
  }
]