import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import udf
from snowflake.snowpark.types import StringType, IntegerType
import re

def main(session: snowpark.Session):
    session.use_database("ECONOMIC_RAG_DB")
    session.use_schema("RAW_DATA")

    def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> str:
        if not text or len(text.strip()) == 0:
            return ""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return "\n---\n".join(chunks[:8])

    # Register permanent UDF with correct stage
    session.udf.register(
        chunk_text,
        name="CHUNK_TEXT_SNOWPARK",
        is_permanent=True,
        stage_location="@CHUNK_STAGE",          # <-- THIS IS THE FIX
        replace=True,
        return_type=StringType(),
        input_types=[StringType(), IntegerType(), IntegerType()]
    )

    # Create chunks table
    session.sql("""
    CREATE OR REPLACE TABLE ECONOMIC_RAG_DB.CHUNKS.ECONOMIC_CHUNKS (
      CHUNK_ID NUMBER AUTOINCREMENT,
      DOC_ID NUMBER,
      SOURCE VARCHAR,
      COMPANY_OR_ENTITY VARCHAR,
      FILING_TYPE_OR_CATEGORY VARCHAR,
      DATE DATE,
      CHUNK_TEXT VARCHAR,
      CHUNK_INDEX NUMBER
    )
    """).collect()

    # Run chunking (5000 rows for fast test)
    session.sql("""
    INSERT INTO ECONOMIC_RAG_DB.CHUNKS.ECONOMIC_CHUNKS 
      (DOC_ID, SOURCE, COMPANY_OR_ENTITY, FILING_TYPE_OR_CATEGORY, DATE, CHUNK_TEXT, CHUNK_INDEX)
    SELECT 
      DOC_ID,
      SOURCE,
      COMPANY_OR_ENTITY,
      FILING_TYPE_OR_CATEGORY,
      DATE,
      CHUNK_TEXT_SNOWPARK(TEXT_CONTENT, 400, 50) AS CHUNK_TEXT,
      ROW_NUMBER() OVER (PARTITION BY DOC_ID ORDER BY 1) AS CHUNK_INDEX
    FROM ECONOMIC_RAG_DB.RAW_DATA.RAW_ECONOMIC_TEXT
    WHERE LENGTH(TEXT_CONTENT) > 100
    LIMIT 5000
    """).collect()

    # Show results
    result = session.sql("""
        SELECT COUNT(*) as total_chunks, COUNT(DISTINCT DOC_ID) as docs_chunked 
        FROM ECONOMIC_RAG_DB.CHUNKS.ECONOMIC_CHUNKS
    """).collect()
    print(result)
    return "✅ Chunking completed successfully!"