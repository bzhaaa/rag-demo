# Corrective RAG Agent

A corrective Retrieval-Augmented Generation (RAG) demo implemented with
LangGraph, OpenAI-compatible model APIs, and a local Milvus vector database.

## Features

- Loads model configuration from `.env`
- Supports separate OpenAI-compatible chat and embedding endpoints
- Supports optional LangSmith tracing for the full LangGraph workflow
- Stores and retrieves document vectors from local Milvus
- Grades retrieved document relevance before generation
- Rewrites weak queries before the fallback branch
- Uses deterministic mock web-search results without external search requests
- Provides a Streamlit UI for local PDF, TXT, and Markdown uploads

## Environment Configuration

The following `.env` values are required:

```dotenv
LLM_API_KEY=your-chat-api-key
LLM_BASE_URL=https://example.com/v1
LLM_MODEL=your-chat-model

EMBEDDING_API_KEY=your-embedding-api-key
EMBEDDING_BASE_URL=https://example.com/v1
EMBEDDING_MODEL=your-embedding-model
```

URLs ending in `/chat/completions` or `/embeddings` are normalized to the API
root automatically.

Optional LangSmith tracing:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=corrective-rag
LANGSMITH_HIDE_INPUTS=false
LANGSMITH_HIDE_OUTPUTS=false
# LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

When enabled, each question creates a `corrective-rag-workflow` trace with
LangGraph node runs, model calls, tags, and model/vector-store metadata.
Set `LANGSMITH_HIDE_INPUTS` or `LANGSMITH_HIDE_OUTPUTS` to `true` when document
content or generated answers must not be included in traces.

Optional local settings:

```dotenv
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=corrective_rag
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run corrective_rag.py
```

## Workflow

1. Upload a local PDF, TXT, or Markdown document.
2. Split the document into overlapping chunks.
3. Embed chunks with the configured embedding endpoint.
4. Store vectors in local Milvus.
5. Retrieve relevant chunks for the user question.
6. Grade retrieved chunks with the configured chat model.
7. If local context is weak, rewrite the query and add a mock search document.
8. Generate the final answer with the configured chat model.

The mock search document is explicitly marked as synthetic and should not be
treated as verified external evidence.
