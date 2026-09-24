# Support Assistant Module

# Module 3 — Zepto Support Assistant (`/support_assistant`)

An offline-first, RAG-enabled customer support assistant service built for Zepto.

---

## 1. Pipeline Architecture Walkthrough

```text
[Customer Query]
       │
       ▼
┌─────────────────────────┐
│  1. classify_intent     │ (Keyword Heuristic / Mock Mode)
└───────────┬─────────────┘
            │
  ┌─────────┴────────────────────────┐
  ▼                                  ▼
[policy_question]            [general_question]
  │                                  │
  ▼                                  ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│ 2. retrieve_and_answer  │  │    3. direct_answer    │
│  - ChromaDB Embeddings  │  │  - Static response      │
│  - Cosine Search (top-3)│  │    (no retrieval)       │
│  - Formats output       │  └───────────┬─────────────┘
└───────────┬─────────────┘              │
            │                            │
            └─────────────┬──────────────┘
                          ▼
             [Pydantic JSON Response]
