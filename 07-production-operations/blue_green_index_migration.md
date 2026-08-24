# Blue-Green Index Migration

## Goal

Replace a production vector index without breaking live RAG traffic.

## Problem

Suppose the current index uses:

- embedding model v1
- chunk size 500
- old metadata schema

A new index is built using:

- embedding model v2
- improved chunking
- new metadata

Updating the existing production index in place is risky.

## Blue-Green Approach

Current production index:

`rag-index-blue`

New index:

`rag-index-green`

Build and validate the green index while users continue querying blue.

```text
Users
  |
  v
rag-index-blue
      |
      | build in parallel
      v
rag-index-green