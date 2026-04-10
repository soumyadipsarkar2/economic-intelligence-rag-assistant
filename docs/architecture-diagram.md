# Architecture diagram (export-ready)

If your README renderer supports Mermaid, you can copy/paste this diagram.
Otherwise, export it to an image (PNG/SVG) and save as `docs/architecture-diagram.png` (or `.svg`) for easy embedding.

```mermaid
flowchart LR
  subgraph Snowflake["Snowflake Account"]
    subgraph RAW["ECONOMIC_RAG_DB.RAW_DATA"]
      RAWT["RAW_ECONOMIC_TEXT\n(SOURCE, ENTITY, CATEGORY, DATE, TEXT_CONTENT)"]
    end

    subgraph CH["ECONOMIC_RAG_DB.CHUNKS"]
      UDF["Snowpark Permanent UDF\nCHUNK_TEXT_SNOWPARK (@CHUNK_STAGE)"]
      CHUNKS["ECONOMIC_CHUNKS\n(CHUNK_TEXT + attributes)"]
    end

    CSS["Cortex Search Service\nECONOMIC_RAG_SEARCH\n(index: CHUNK_TEXT)"]
    SEARCH["SNOWFLAKE.CORTEX.SEARCH_PREVIEW\n(top‑k JSON results)"]
    LLM["SNOWFLAKE.CORTEX.COMPLETE\n(model: llama3.1-70b)\n(grounded answer + Sources)"]

    subgraph APP["Streamlit in Snowflake"]
      UI["Economic Intelligence Chat UI\n(conversation history)"]
    end

    subgraph EVAL["Evaluation"]
      GT["EVAL_GROUND_TRUTH\n(20 questions)"]
      SP["RUN_RAG_EVALUATION()\n(Python stored procedure)"]
      RES["EVAL_RESULTS\n(generated answers)"]
      REP["Precision Summary Query\n(heuristic match)"]
    end
  end

  RAWT --> UDF --> CHUNKS --> CSS --> SEARCH --> LLM --> UI
  GT --> SP --> RES --> REP
  SP --> SEARCH
  SP --> LLM
```

