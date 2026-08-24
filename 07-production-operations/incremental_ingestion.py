"""
Goal:
Show how incremental ingestion updates only documents
that changed since the last indexing run.
"""

documents = [
    {"id": "doc-1", "updated_at": "2026-08-20", "text": "Glue retry guidance."},
    {"id": "doc-2", "updated_at": "2026-08-24", "text": "Updated throttling guidance."},
    {"id": "doc-3", "updated_at": "2026-08-18", "text": "Glue worker configuration."},
]

last_indexed_at = "2026-08-22"

# ISO 8601 dates ("YYYY-MM-DD") sort correctly as plain strings, so this
# comparison works without parsing into datetime objects. Only doc-2
# qualifies here: doc-1 and doc-3 were last updated before the watermark
# and would be wasted work to re-embed and re-index.
changed_documents = [
    doc
    for doc in documents
    if doc["updated_at"] > last_indexed_at
]

print(f"Last indexed at: {last_indexed_at}\n")

print("Documents to re-index:")
for doc in changed_documents:
    print(f"- {doc['id']} | updated={doc['updated_at']} | {doc['text']}")

print(
    "\nLesson: Incremental ingestion processes only new or changed "
    "documents, reducing indexing time and cost."
)