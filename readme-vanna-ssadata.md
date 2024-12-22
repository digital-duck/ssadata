

## main changes

- vanna/utils.py
- openai/openai_chat.py
- google/gemini_chat.py
- anthropic/anthropic_chat.py
- bedrock/bedrock_converse.py
- ollama/ollama.py

- vanna/base/base.py
    - major changes:
        - class AskResult
        - class LogTag
        - SQL_DIALECTS
        - VECTOR_DB_LIST
        - add dataset level collection, remove ddl/sql/doc level collections, treat them as metadata
        - def train: add dataset to API
        - def ask_adaptive()
        - add def ask_llm()
        - refactor def ask() to return AskResult


in `vanna/chromadb/chromadb_vector.py

replace the following 3 collections with 1 collection
named after dataset, old APIs added "_nods" suffix
```
add_question_sql
add_ddl
add_documentation
```