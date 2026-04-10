-- =============================================
-- FILE: 04_EVALUATION_20_QUESTIONS.sql
-- Step 4: Ground-truth evaluation (20 Q&A pairs + report)
-- =============================================

USE DATABASE ECONOMIC_RAG_DB;
USE SCHEMA CHUNKS;

-- 1. Create ground-truth table (20 questions)
CREATE OR REPLACE TABLE EVAL_GROUND_TRUTH (
    QUESTION_ID NUMBER AUTOINCREMENT,
    QUESTION VARCHAR,
    EXPECTED_ANSWER VARCHAR,
    EXPECTED_ENTITIES VARCHAR
);

INSERT INTO EVAL_GROUND_TRUTH (QUESTION, EXPECTED_ANSWER, EXPECTED_ENTITIES) VALUES
('What is the measure for civilian labor force?', 'Civilian Labor Force', 'Civilian Labor Force'),
('What is the unit for unemployment rate?', 'Percent', 'Unemployment Rate'),
('What is PCE Price Index for energy goods and services?', 'Price Index (Percent change, year ago)', 'Energy goods and services'),
('What is the frequency of Current Employment Statistics?', 'Monthly', 'Current Employment Statistics (National)'),
('What is seasonally adjusted data?', 'Indicates whether the value is seasonally adjusted', 'Seasonally Adjusted'),
('What is the measure for personal consumption expenditures?', 'Personal consumption expenditures (PCE)', 'PCE'),
('What is the unit for average weekly earnings?', 'USD per week', 'Average weekly earnings'),
('What industry has testing laboratories and services?', 'Testing Laboratories And Services', 'Testing Laboratories And Services'),
('What is the release name for PCE Price Index?', 'Personal Consumption Expenditures Price Index', 'Personal Consumption Expenditures Price Index'),
('What is the definition of VARIABLE_NAME?', 'Human-readable unique name for the variable.', 'VARIABLE_NAME'),
('What is the frequency for annual labor force data?', 'Annual', 'Annual'),
('What is the measure for unemployment?', 'Unemployment', 'Unemployment'),
('What is the unit for labor force participation rate?', 'Percent', 'Labor Force Participation'),
('What is the release for Current Population Survey?', 'Current Population Survey (CPS)', 'Current Population Survey (CPS)'),
('What does MEASUREMENT_TYPE describe?', 'Details how the variable was measured or calculated', 'MEASUREMENT_TYPE'),
('What is the industry for general medical and surgical hospitals?', 'General Medical And Surgical Hospitals', 'General Medical And Surgical Hospitals'),
('What is the seasonally adjusted status for most monthly data?', 'Seasonally adjusted or Not seasonally adjusted', 'SEASONALLY_ADJUSTED'),
('What is the unit for count data in employment?', 'Count', 'Count'),
('What is the measure for price index?', 'Price Index', 'Price Index'),
('What is the definition of RELEASE_NAME?', 'The individual report or collection of data in which the data was released.', 'RELEASE_NAME');

-- 2. Python Stored Procedure (fixed - no backslash in f-string)
CREATE OR REPLACE PROCEDURE RUN_RAG_EVALUATION()
RETURNS VARCHAR
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'run_evaluation'
AS
$$
import snowflake.snowpark as snowpark

def run_evaluation(session: snowpark.Session):
    # Create results table
    session.sql("""
        CREATE OR REPLACE TABLE EVAL_RESULTS (
            QUESTION_ID NUMBER,
            QUESTION VARCHAR,
            EXPECTED_ANSWER VARCHAR,
            EXPECTED_ENTITIES VARCHAR,
            GENERATED_ANSWER VARCHAR
        )
    """).collect()

    # Get all questions
    questions = session.sql("""
        SELECT QUESTION_ID, QUESTION, EXPECTED_ANSWER, EXPECTED_ENTITIES 
        FROM EVAL_GROUND_TRUTH 
        ORDER BY QUESTION_ID
    """).collect()

    for row in questions:
        qid = row['QUESTION_ID']
        question = row['QUESTION']
        expected = row['EXPECTED_ANSWER']
        entities = row['EXPECTED_ENTITIES']

        # Escape question safely BEFORE f-string
        escaped_question = question.replace('"', '\\"')
        json_payload = f'{{"query": "{escaped_question}", "limit": 8}}'

        # Get search results as JSON
        search_result = session.sql(f"""
            SELECT SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
                'ECONOMIC_RAG_SEARCH', 
                '{json_payload}'
            ) AS JSON_RESULT
        """).collect()[0]['JSON_RESULT']

        # Build context in Python (simple)
        import json
        results = json.loads(search_result).get('results', [])
        context = ""
        for r in results:
            context += f"SOURCE: {r.get('ATTRIBUTES', {}).get('SOURCE', '')}\n"
            context += f"ENTITY: {r.get('ATTRIBUTES', {}).get('COMPANY_OR_ENTITY', '')}\n"
            context += f"TEXT: {r.get('CHUNK_TEXT', '')}\n---\n"

        # Generate answer
        prompt = f"""You are an expert economic intelligence assistant.

Use ONLY the context below to answer the question.
If you cannot answer from the context, say "I cannot find sufficient information in the dataset."

=== CONTEXT ===
{context}
=== QUESTION ===
{question}

Answer in clear, professional English.
At the very end add a section called "Sources:"."""

        generated = session.sql(f"""
            SELECT SNOWFLAKE.CORTEX.COMPLETE('llama3.1-70b', '{prompt.replace("'", "''")}') AS ANSWER
        """).collect()[0]['ANSWER']

        # Insert result
        session.sql(f"""
            INSERT INTO EVAL_RESULTS 
            VALUES ({qid}, '{question.replace("'", "''")}', 
                    '{expected.replace("'", "''")}', 
                    '{entities.replace("'", "''")}', 
                    '{generated.replace("'", "''")}')
        """).collect()

    return "✅ Evaluation completed successfully! Check EVAL_RESULTS table."
$$
;
-- Create empty results table first
CREATE OR REPLACE TABLE EVAL_RESULTS (
    QUESTION_ID NUMBER,
    QUESTION VARCHAR,
    EXPECTED_ANSWER VARCHAR,
    EXPECTED_ENTITIES VARCHAR,
    GENERATED_ANSWER VARCHAR
);

-- Call the procedure
CALL RUN_RAG_EVALUATION();

-- View the results + report
SELECT * FROM EVAL_RESULTS ORDER BY QUESTION_ID;

-- Evaluation summary
SELECT 
    'Evaluation Summary' AS METRIC,
    COUNT(*) AS TOTAL_QUESTIONS,
    SUM(CASE WHEN CONTAINS(GENERATED_ANSWER, EXPECTED_ANSWER) OR CONTAINS(GENERATED_ANSWER, EXPECTED_ENTITIES) THEN 1 ELSE 0 END) AS CORRECT_ANSWERS,
    ROUND(100.0 * SUM(CASE WHEN CONTAINS(GENERATED_ANSWER, EXPECTED_ANSWER) OR CONTAINS(GENERATED_ANSWER, EXPECTED_ENTITIES) THEN 1 ELSE 0 END) / COUNT(*), 2) AS RETRIEVAL_PRECISION_PCT
FROM EVAL_RESULTS;