# ARCHITECTURE — Economic Intelligence RAG Assistant (Snowflake-native)

This document describes the end-to-end architecture for a **RAG-powered Economic Intelligence Assistant built entirely in Snowflake**, including the data model, retrieval/indexing, grounded generation, Streamlit experience, and evaluation pipeline.

---

## Goals and constraints

- **Snowflake-only**: No external vector DB, no external LLM calls, no separate embedding service.
- **Grounded answers**: The assistant must answer using retrieved evidence; otherwise it must decline.
- **Traceability**: Every answer includes a `Sources:` section derived from retrieval attributes.
- **Repeatable evaluation**: A small but complete evaluation suite is provided and can run end-to-end inside Snowflake.

---

## Core components

### 1) Storage & schemas

Database: `ECONOMIC_RAG_DB`

- **Schema `RAW_DATA`**
  - `RAW_ECONOMIC_TEXT`
    - Purpose: normalized “document-like” table where each row is a source text object to be chunked
    - Key columns:
      - `SOURCE`: dataset/source identifier
      - `COMPANY_OR_ENTITY`: entity name / variable / subject
      - `FILING_TYPE_OR_CATEGORY`: release name / category
      - `DATE`: optional date
      - `TEXT_CONTENT`: long-form text built by concatenating descriptive fields

- **Schema `CHUNKS`**
  - `ECONOMIC_CHUNKS`
    - Purpose: retrieval-ready chunk table indexed by Cortex Search
    - Key columns:
      - `CHUNK_ID`: unique chunk identifier
      - `DOC_ID`: source document id (row id from `RAW_ECONOMIC_TEXT`)
      - `CHUNK_TEXT`: chunk body used for indexing and context
      - `CHUNK_INDEX`: within-document order (for debugging and traceability)
      - attributes copied from raw: `SOURCE`, `COMPANY_OR_ENTITY`, `FILING_TYPE_OR_CATEGORY`, `DATE`

Warehouse:

- `RAG_WH` (used by Cortex Search Service and heavy SQL)

Stages:

- `@CHUNK_STAGE` (used to store the permanent Snowpark UDF dependency package for chunking)

---

## Data flow (end-to-end)

### Step A — Create environment

Script: `01_CREATE_DATABASE_AND_SCHEMAS.sql`

- creates `ECONOMIC_RAG_DB`, schemas, and `RAG_WH`
- provides helper SQL to locate relevant public/marketplace tables
- inserts a sample “economic text corpus” into `RAW_ECONOMIC_TEXT` by concatenating descriptive columns

### Step B — Chunking (Snowpark UDF)

Script: `01_CHUNKING_PYTHON.py`

- registers a permanent UDF:
  - name: `CHUNK_TEXT_SNOWPARK`
  - stage: `@CHUNK_STAGE`
- creates `ECONOMIC_RAG_DB.CHUNKS.ECONOMIC_CHUNKS`
- inserts chunked rows into `ECONOMIC_CHUNKS`

Chunking behavior (current implementation):

- splits text into sentence-ish segments using a regex boundary
- accumulates into chunks up to ~`chunk_size` characters
- returns up to the first ~8 chunks joined with `---` separators

Operational note:

- The chunking implementation is intentionally simple for hackathon reproducibility; in production, you would emit one row per chunk (rather than joining) and tune chunk/overlap to optimize retrieval.

### Step C — Retrieval indexing (Cortex Search Service)

Script: `02_CREATE_CORTEX_SEARCH_SERVICE.sql`

Creates:

- `CORTEX SEARCH SERVICE ECONOMIC_RAG_SEARCH`
  - **Index column**: `CHUNK_TEXT`
  - **Attributes**: `SOURCE`, `COMPANY_OR_ENTITY`, `FILING_TYPE_OR_CATEGORY`, `DATE`
  - **Warehouse**: `RAG_WH`
  - **Freshness**: `TARGET_LAG = '5 minutes'`

The service enables semantic retrieval using Snowflake-managed embeddings (Cortex Search).

### Step D — Grounded generation (RAG query)

Script: `03_RAG_PIPELINE.sql`

1) Retrieval:

- Uses `SNOWFLAKE.CORTEX.SEARCH_PREVIEW('ECONOMIC_RAG_SEARCH', '{"query": "...", "limit": 8}')`
- Parses returned JSON and flattens `results` to obtain:
  - `CHUNK_TEXT`
  - attribute fields used later for citations

2) Prompt assembly:

- Concatenates retrieved chunks into a context block:
  - includes `SOURCE`, `ENTITY`, `DATE`, and the `TEXT`

3) Completion:

- `SNOWFLAKE.CORTEX.COMPLETE('llama3.1-70b', prompt)`
- Prompt constraints:
  - “Use ONLY the context below…”
  - “If the answer is not in the context, say …”
  - “Add a Sources: section at the end…”

### Step E — Streamlit chat experience

App: `05_ECONOMIC_INTELLIGENCE_CHAT.streamlit`

Runtime behavior:

- Maintains chat history in `st.session_state.messages`
- On each user prompt:
  1. calls `SEARCH_PREVIEW` to retrieve top-k chunks
  2. builds a context string in Python
  3. calls `CORTEX.COMPLETE` with the grounded prompt
  4. renders the answer and appends to chat history

Notes:

- The sidebar is reserved for citations UX. The current version includes a citations header; a natural next enhancement is to parse retrieval results into a compact, clickable citation list.

### Step F — Evaluation (20 questions)

Script: `04_EVALUATION_20_QUESTIONS.sql`

Creates:

- `EVAL_GROUND_TRUTH`: 20 questions, expected answer string, expected entity string
- `EVAL_RESULTS`: stores generated answers
- `RUN_RAG_EVALUATION()`:
  - loops through questions
  - runs retrieval
  - builds context
  - runs `CORTEX.COMPLETE`
  - inserts results into `EVAL_RESULTS`

Summary query:

- counts a question as correct when the generated answer contains the expected answer **or** expected entity string
- produces `RETRIEVAL_PRECISION_PCT` (heuristic, judge-friendly)

---

## Interfaces & contracts

### Retrieval contract (`ECONOMIC_RAG_SEARCH`)

Inputs:

- `query` (string)
- `limit` (int)

Outputs (JSON):

- `results[]` where each result contains:
  - `CHUNK_TEXT`
  - `ATTRIBUTES` (including `SOURCE`, `COMPANY_OR_ENTITY`, optional `DATE`, etc.)

### Generation contract (grounded completion)

Inputs:

- Prompt that includes:
  - an explicit “ONLY use context” instruction
  - the retrieved context block
  - the user question

Output:

- A natural-language answer that ends with:
  - `Sources:` list (derived from the context)

---

## Security & governance notes (hackathon-ready)

- **Data locality**: All processing stays within Snowflake.
- **Access control**: Prefer granting to a dedicated role (the scripts currently show `PUBLIC` for simplicity).
- **Cost controls**: `RAG_WH` is set to auto-suspend; Cortex usage should be monitored in account usage views.

---

## Scaling and next improvements

- **Chunk table normalization**: Emit one row per chunk rather than joining chunks into a single string.
- **Better evaluation**:
  - exact-match + semantic scoring
  - retrieval recall metrics (does retrieved context contain gold entity?)
- **Citations UX**: Show citations as structured cards linking `SOURCE/ENTITY/DATE` to chunk text snippets.
- **Filters**: Add optional attribute filters (e.g., `SOURCE` or date ranges) at retrieval time.

