-- =============================================
-- FILE: 03_RAG_PIPELINE.sql
-- Step 3: Working RAG pipeline (using official SEARCH_PREVIEW)
-- =============================================

USE DATABASE ECONOMIC_RAG_DB;
USE SCHEMA CHUNKS;

-- 1. Simple test (run this first — must return results)
SELECT 
    PARSE_JSON(
        SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
            'ECONOMIC_RAG_SEARCH', 
            '{"query": "test query", "limit": 3}'
        )
    ) AS PREVIEW_RESULTS;

-- 2. FULL RAG PIPELINE with citations (this is the final working query)
WITH RAW_RESULTS AS (
    SELECT 
        PARSE_JSON(
            SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
                'ECONOMIC_RAG_SEARCH',
                '{"query": "What is the unemployment situation for women with children?", "limit": 8}'
            )
        ) AS JSON_RESULT
),
FLATTENED AS (
    SELECT 
        VALUE:ATTRIBUTES:SOURCE::STRING AS SOURCE,
        VALUE:ATTRIBUTES:COMPANY_OR_ENTITY::STRING AS COMPANY_OR_ENTITY,
        VALUE:ATTRIBUTES:DATE::STRING AS DATE,
        VALUE:CHUNK_TEXT::STRING AS CHUNK_TEXT
    FROM RAW_RESULTS,
    TABLE(FLATTEN(INPUT => JSON_RESULT:results)) 
)
SELECT 
    SNOWFLAKE.CORTEX.COMPLETE(
        'llama3.1-70b',
        'You are an expert economic intelligence assistant.\n\n' ||
        'Use ONLY the context below to answer the question.\n' ||
        'If the answer is not in the context, say "I cannot find sufficient information in the dataset."\n\n' ||
        '=== CONTEXT START ===\n' ||
        (SELECT LISTAGG(
            'SOURCE: ' || SOURCE || '\n' ||
            'ENTITY: ' || COMPANY_OR_ENTITY || '\n' ||
            'DATE: ' || COALESCE(DATE, 'N/A') || '\n' ||
            'TEXT: ' || CHUNK_TEXT || '\n---\n',
            ''
        ) FROM FLATTENED) ||
        '=== CONTEXT END ===\n\n' ||
        'Question: What is the unemployment situation for women with children?\n\n' ||
        'Answer in clear, professional English.\n' ||
        'At the very end of your answer, add a section called "Sources:" and list the exact entities/sources you used.'
    ) AS GROUNDED_ANSWER_WITH_CITATIONS;