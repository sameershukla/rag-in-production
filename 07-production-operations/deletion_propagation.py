"""
Goal:
Show how deleting a source document must also remove
its chunks from the retrieval index.
"""

# A single source document is split into multiple chunks (doc-1 has two),
# so deleting "doc-1" has to be resolved through document_id -- deleting by
# chunk_id alone would miss sibling chunks and leave stale ones behind.
vector_index = [
    {"chunk_id": "doc-1-1", "document_id": "doc-1", "text": "Glue overview."},
    {"chunk_id": "doc-1-2", "document_id": "doc-1", "text": "Glue retries."},
    {"chunk_id": "doc-2-1", "document_id": "doc-2", "text": "Payroll runbook."},
    {"chunk_id": "doc-3-1", "document_id": "doc-3", "text": "S3 guidance."},
]

deleted_document_id = "doc-2"

print("Before deletion:")
for chunk in vector_index:
    print(f"- {chunk['chunk_id']} | {chunk['document_id']} | {chunk['text']}")

# Every chunk sharing the deleted document_id must go, e.g. both doc-1-1
# and doc-1-2 for doc-1 -- a partial removal would still let the retriever
# surface leftover fragments of a document that's supposed to be gone.
vector_index = [
    chunk
    for chunk in vector_index
    if chunk["document_id"] != deleted_document_id
]

print(f"\nDeleted source document: {deleted_document_id}")

print("\nAfter deletion propagation:")
for chunk in vector_index:
    print(f"- {chunk['chunk_id']} | {chunk['document_id']} | {chunk['text']}")

print(
    "\nLesson: Source deletions must propagate to every indexed chunk, "
    "otherwise stale or unauthorized content can still be retrieved."
)