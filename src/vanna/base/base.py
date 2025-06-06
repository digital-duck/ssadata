r"""

# Nomenclature

| Prefix | Definition | Examples |
| --- | --- | --- |
| `vn.get_` | Fetch some data | [`vn.get_related_ddl(...)`][vanna.base.base.VannaBase.get_related_ddl] |
| `vn.add_` | Adds something to the retrieval layer | [`vn.add_question_sql(...)`][vanna.base.base.VannaBase.add_question_sql] <br> [`vn.add_ddl(...)`][vanna.base.base.VannaBase.add_ddl] |
| `vn.generate_` | Generates something using AI based on the information in the model | [`vn.generate_sql(...)`][vanna.base.base.VannaBase.generate_sql] <br> [`vn.generate_explanation()`][vanna.base.base.VannaBase.generate_explanation] |
| `vn.run_` | Runs code (SQL) | [`vn.run_sql`][vanna.base.base.VannaBase.run_sql] |
| `vn.remove_` | Removes something from the retrieval layer | [`vn.remove_training_data`][vanna.base.base.VannaBase.remove_training_data] |
| `vn.connect_` | Connects to a database | [`vn.connect_to_snowflake(...)`][vanna.base.base.VannaBase.connect_to_snowflake] |
| `vn.update_` | Updates something | N/A -- unused |
| `vn.set_` | Sets something | N/A -- unused  |

# Open-Source and Extending

Vanna.AI is open-source and extensible. If you'd like to use Vanna without the servers, see an example [here](https://vanna.ai/docs/postgres-ollama-chromadb/).

The following is an example of where various functions are implemented in the codebase when using the default "local" version of Vanna. `vanna.base.VannaBase` is the base class which provides a `vanna.base.VannaBase.ask` and `vanna.base.VannaBase.train` function. Those rely on abstract methods which are implemented in the subclasses `vanna.openai_chat.OpenAI_Chat` and `vanna.chromadb_vector.ChromaDB_VectorStore`. `vanna.openai_chat.OpenAI_Chat` uses the OpenAI API to generate SQL and Plotly code. `vanna.chromadb_vector.ChromaDB_VectorStore` uses ChromaDB to store training data and generate embeddings.

If you want to use Vanna with other LLMs or databases, you can create your own subclass of `vanna.base.VannaBase` and implement the abstract methods.

```mermaid
flowchart
    subgraph VannaBase
        ask
        train
    end

    subgraph OpenAI_Chat
        get_sql_prompt
        submit_prompt
        generate_question
        generate_plotly_code
    end

    subgraph ChromaDB_VectorStore
        generate_embedding
        add_question_sql
        add_ddl
        add_documentation
        get_similar_question_sql
        get_related_ddl
        get_related_documentation
    end
```

"""

import json
import os
import sys
import re
import sqlite3
import time
import traceback
from abc import ABC, abstractmethod
import logging
from typing import List, Tuple, Union, NamedTuple, Optional
from urllib.parse import urlparse

import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objs as Figure
import requests
import sqlparse

from contextlib import contextmanager

@contextmanager
def suppress_warnings_only():
    """
        Suppress warning messages from ChromaDB
    """
    class WarningFilter:
        FILTERED_MESSAGES = [
            "Number of requested results",
            "Insert of existing embedding ID",
            "Add of existing embedding ID",
        ]
        
        def write(self, x): 
            # Only write if none of the filtered messages are in x
            if not any(msg in x for msg in self.FILTERED_MESSAGES):
                old_stderr.write(x)
                
        def flush(self): 
            pass
    
    old_stderr = sys.stderr
    sys.stderr = WarningFilter()
    try:
        yield
    finally:
        sys.stderr = old_stderr

from ..exceptions import DependencyError, ImproperlyConfigured, ValidationError
from ..types import TrainingPlan, TrainingPlanItem, TableMetadata
from ..utils import (
    SEPARATOR,
    vn_log,
    validate_config_path, 
    strip_brackets, 
    remove_sql_noise,
    take_last_n_messages,
    # extract_sql  # To verify
)

try:
    from IPython.display import display, Code, Image
    HAS_IPYTHON = True
except Exception as e:
    print(f"Failed to import IPython: {str(e)}")
    HAS_IPYTHON = False

PREFIX_MY = "my-"

class AskResult(NamedTuple):
    sql: Optional[Tuple[Optional[str], float, Optional[str]]]
    df: Optional[Tuple[Optional[pd.DataFrame], float, Optional[str]]]
    py: Optional[Tuple[Optional[str], float, Optional[str]]]
    fig: Optional[Tuple[Optional[Figure.Figure], float, Optional[str]]]
    has_error: bool

class LogTag:
    ERROR = "[ERROR]"
    ERROR_INPUT = "[ERROR-IN]"
    ERROR_SQL = "[ERROR-SQL]"
    ERROR_DB = "[ERROR-DB]"
    ERROR_DF = "[ERROR-DF]"
    ERROR_VIZ = "[ERROR-VIZ]"
    CTX_PROMPT = "Context PROMPT"
    SQL_PROMPT = "SQL PROMPT"
    SHOW_LLM = "<LLM>"
    SHOW_CXT = "<Context>"
    SHOW_DATA = "<DataFrame>"
    SHOW_SQL = "<SQL>"
    SHOW_PYTHON = "<Python>"
    SHOW_VIZ = "<Chart>"
    LLM_RESPONSE = "LLM RESPONSE"
    RUN_INTER_SQL = "INTERMEDIATE SQL"
    EXTRACTED_SQL = "EXTRACTED SQL"
    RETRY = "RETRY"
    
def collect_err_msg(answer):
    err_msg = ""
    has_error_sql = has_error_df = has_error_py = has_error_fig = False
    if answer.sql and answer.sql[2]:
        has_error_sql = LogTag.ERROR_DB in answer.sql[2]
        err_msg += answer.sql[2]
    if answer.df and answer.df[2]:
        has_error_df = LogTag.ERROR_DF in answer.df[2]
        err_msg += answer.df[2]
    if answer.py and answer.py[2]:
        has_error_py = LogTag.ERROR_VIZ in answer.py[2]
        err_msg += answer.py[2]
    if answer.fig and answer.fig[2]:
        has_error_fig = LogTag.ERROR_DF in answer.fig[2]
        err_msg += answer.fig[2]
    return err_msg, any((has_error_sql, has_error_df, has_error_py, has_error_fig))

#=================================
# helper functions
#=================================
def keep_latest_messages(prompt_json):
    latest_messages = {}
    
    for message in reversed(prompt_json):
        role = message['role']
        if role not in latest_messages:
            latest_messages[role] = message
    
    return [latest_messages[role] for role in ['system', 'assistant', 'user'] if role in latest_messages]

def skip_chart(question):
    x = question.lower()
    if ("skip" in x or "no" in x) and "chart" in x:
        return True
    
    return False
#=================================
# main functionality
#=================================
class VannaBase(ABC):

    SQL_DIALECTS = [
        "Snowflake",
        "SQLite",
        "PostgreSQL",
        "MySQL",
        "ClickHouse",
        "Oracle",
        "BigQuery",
        "DuckDB",
        "Microsoft SQL Server",
        "Presto",
        "Hive",
    ]

    VECTOR_DB_LIST = [
        "chromadb", 
        "marqo", 
        "opensearch", 
        "pinecone", 
        "qdrant",
        "faiss",
        "milvus",
        "pgvector",
        "weaviate",
    ]

    def __init__(self, config=None):
        if config is None:
            config = {}

        self.config = config
        self.run_sql_is_set = False
        self.static_documentation = ""
        self.dialect = self.config.get("dialect", "SQL")
        self.language = self.config.get("language", None)
        self.max_tokens = self.config.get("max_tokens", 14000)

    def log(self, message: str, title: str = "", off_flag: bool = False):
        vn_log(message, title, off_flag)

    def _response_language(self) -> str:
        if self.language is None:
            return ""

        return f"Respond in the {self.language} language."

    def summarize_context(self, question: str, print_prompt=True, print_response=True, **kwargs) -> str:
        if self.config is not None:
            initial_prompt = self.config.get("initial_prompt", None)
        else:
            initial_prompt = None
        question_sql_list = self.get_similar_question_sql(question, **kwargs)
        ddl_list = self.get_related_ddl(question, **kwargs)
        doc_list = self.get_related_documentation(question, **kwargs)
        prompt = self.get_context_prompt(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            **kwargs,
        )

        vn_log(title=LogTag.CTX_PROMPT, message=prompt, off_flag=not print_prompt)
        llm_response = self.submit_prompt(prompt, print_prompt=print_prompt, print_response=print_response, **kwargs)
        vn_log(title=LogTag.LLM_RESPONSE, message=llm_response, off_flag=not print_response)

        return llm_response

    def get_context_prompt(
        self,
        initial_prompt : str,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        **kwargs,
    ):
        if initial_prompt is None:
            initial_prompt = f"You are an AI assistant with extensive database knowledge and capable of answering data-related questions. " + \
            "Your response should ONLY be based on the given context and follow the response guidelines and format instructions. "

        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        initial_prompt += (
            "===Response Guidelines \n"
            "1. If the provided context is sufficient, please give a summary answer by synthesizing the context and prompt. \n"
            "2. If the provided context is insufficient, please explain why it can't be generated and say I don't know. \n"
        )

        message_log = [self.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is None:
                print("example is None")
            else:
                if example is not None and "question" in example and "sql" in example:
                    message_log.append(self.user_message(example["question"]))
                    message_log.append(self.assistant_message(example["sql"]))

        message_log.append(self.user_message(question))

        return message_log


    def generate_sql(self, question: str, allow_llm_to_see_data=False, 
                     print_prompt=True, print_response=True, 
                     use_last_n_message=1, 
                     **kwargs) -> str:
        """
        Example:
        ```python
        vn.generate_sql("What are the top 10 customers by sales?")
        ```

        Uses the LLM to generate a SQL query that answers a question. It runs the following methods:

        - [`get_similar_question_sql`][vanna.base.base.VannaBase.get_similar_question_sql]

        - [`get_related_ddl`][vanna.base.base.VannaBase.get_related_ddl]

        - [`get_related_documentation`][vanna.base.base.VannaBase.get_related_documentation]

        - [`get_sql_prompt`][vanna.base.base.VannaBase.get_sql_prompt]

        - [`submit_prompt`][vanna.base.base.VannaBase.submit_prompt]


        Args:
            question (str): The question to generate a SQL query for.
            allow_llm_to_see_data (bool): Whether to allow the LLM to see the data (for the purposes of introspecting the data to generate the final SQL).

        Returns:
            str: The SQL query that answers the question.
        """
        if self.config is not None:
            initial_prompt = self.config.get("initial_prompt", None)
        else:
            initial_prompt = None


        question_sql_list = self.get_similar_question_sql(question, **kwargs)
        ddl_list = self.get_related_ddl(question, **kwargs)
        doc_list = self.get_related_documentation(question, **kwargs)
        prompt = self.get_sql_prompt(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            **kwargs,
        )


        prompt = take_last_n_messages(prompt, n=use_last_n_message)

        vn_log(title=LogTag.SQL_PROMPT, message=prompt, off_flag=not print_prompt)
        llm_response = self.submit_prompt(prompt, print_prompt=print_prompt, print_response=print_response, **kwargs)
        vn_log(title=LogTag.LLM_RESPONSE, message=llm_response, off_flag=not print_response)

        if 'intermediate_sql' in llm_response:
            if not allow_llm_to_see_data:
                return "The LLM is not allowed to see the data in your database. Your question requires database introspection to generate the necessary SQL. Please set allow_llm_to_see_data=True to enable this."

            if allow_llm_to_see_data:
                intermediate_sql = self.extract_sql(llm_response)

                try:
                    vn_log(title=LogTag.RUN_INTER_SQL, message=intermediate_sql, off_flag=not print_response)
                    df = self.run_sql(intermediate_sql)

                    prompt = self.get_sql_prompt(
                        initial_prompt=initial_prompt,
                        question=question,
                        question_sql_list=question_sql_list,
                        ddl_list=ddl_list,
                        doc_list=doc_list+[f"The following is a pandas DataFrame with the results of the intermediate SQL query {intermediate_sql}: \n" + df.to_markdown()],
                        **kwargs,
                    )
                    vn_log(title=LogTag.SQL_PROMPT, message=prompt, off_flag=not print_prompt)
                    llm_response = self.submit_prompt(prompt, **kwargs)
                    vn_log(title=LogTag.LLM_RESPONSE, message=llm_response, off_flag=not print_response)
                except Exception as e:
                    return f"Error running intermediate SQL: {e}"

        extracted_sql = self.extract_sql(llm_response).strip()
        sql_row_limit = int(kwargs.get("sql_row_limit", -1))
        if sql_row_limit > 0 and "limit " not in extracted_sql.lower():
            if extracted_sql[-1] == ";":
                extracted_sql = extracted_sql[:-1]  # remove last ";" if present
            extracted_sql += f" limit {sql_row_limit}"

        return extracted_sql

    def extract_sql(self, llm_response: str) -> str:
        """
        Example:
        ```python
        vn.extract_sql("Here's the SQL query in a code block: ```sql\nSELECT * FROM customers\n```")
        ```

        Extracts the SQL query from the LLM response. This is useful in case the LLM response contains other information besides the SQL query.
        Override this function if your LLM responses need custom extraction logic.

        Args:
            llm_response (str): The LLM response.

        Returns:
            str: The extracted SQL query.
        """

        # If the llm_response contains a markdown code block, with or without the sql tag, extract the last sql from it
        sqls = re.findall(r"```sql\n(.*)```", llm_response, re.IGNORECASE | re.DOTALL)
        if sqls:
            sql = remove_sql_noise(sqls[-1])
            vn_log(title=LogTag.EXTRACTED_SQL, message=f"{sql}")
            return sql

        sqls = re.findall(r"```(.*)```", llm_response, re.DOTALL)
        if sqls:
            sql = remove_sql_noise(sqls[-1])
            vn_log(title=LogTag.EXTRACTED_SQL, message=f"{sql}")
            return sql


        # If the llm_response contains a CTE (with clause), extract the last sql between WITH and ;
        sqls = re.findall(r"\bWITH\b .*?;", llm_response, re.IGNORECASE | re.DOTALL)
        if sqls:
            sql = remove_sql_noise(sqls[-1])
            vn_log(title=LogTag.EXTRACTED_SQL, message=f"{sql}")
            return sql

        # If the llm_response is not markdown formatted, extract last sql by finding select and ; in the response
        sqls = re.findall(r"SELECT.*?;", llm_response, re.IGNORECASE | re.DOTALL)
        if sqls:
            sql = remove_sql_noise(sqls[-1])
            vn_log(title=LogTag.EXTRACTED_SQL, message=f"{sql}")
            return sql


        return llm_response

    def extract_table_metadata(ddl: str) -> TableMetadata:
      """
        Example:
        ```python
        vn.extract_table_metadata("CREATE TABLE hive.bi_ads.customers (id INT, name TEXT, sales DECIMAL)")
        ```

        Extracts the table metadata from a DDL statement. This is useful in case the DDL statement contains other information besides the table metadata.
        Override this function if your DDL statements need custom extraction logic.

        Args:
            ddl (str): The DDL statement.

        Returns:
            TableMetadata: The extracted table metadata.
        """
      pattern_with_catalog_schema = re.compile(
        r'CREATE TABLE\s+(\w+)\.(\w+)\.(\w+)\s*\(',
        re.IGNORECASE
      )
      pattern_with_schema = re.compile(
        r'CREATE TABLE\s+(\w+)\.(\w+)\s*\(',
        re.IGNORECASE
      )
      pattern_with_table = re.compile(
        r'CREATE TABLE\s+(\w+)\s*\(',
        re.IGNORECASE
      )

      match_with_catalog_schema = pattern_with_catalog_schema.search(ddl)
      match_with_schema = pattern_with_schema.search(ddl)
      match_with_table = pattern_with_table.search(ddl)

      if match_with_catalog_schema:
        catalog = match_with_catalog_schema.group(1)
        schema = match_with_catalog_schema.group(2)
        table_name = match_with_catalog_schema.group(3)
        return TableMetadata(catalog, schema, table_name)
      elif match_with_schema:
        schema = match_with_schema.group(1)
        table_name = match_with_schema.group(2)
        return TableMetadata(None, schema, table_name)
      elif match_with_table:
        table_name = match_with_table.group(1)
        return TableMetadata(None, None, table_name)
      else:
        return TableMetadata()

    def is_sql_valid(self, sql: str) -> bool:
        """
        Example:
        ```python
        vn.is_sql_valid("SELECT * FROM customers")
        ```
        Checks if the SQL query is valid. This is usually used to check if we should run the SQL query or not.
        By default it checks if the SQL query is a SELECT statement. You can override this method to enable running other types of SQL queries.

        Args:
            sql (str): The SQL query to check.

        Returns:
            bool: True if the SQL query is valid, False otherwise.
        """

        parsed = sqlparse.parse(sql)

        for statement in parsed:
            if statement.get_type() == 'SELECT':
                return True

        return False

    def should_generate_chart(self, df: pd.DataFrame) -> bool:
        """
        Example:
        ```python
        vn.should_generate_chart(df)
        ```

        Checks if a chart should be generated for the given DataFrame. By default, it checks if the DataFrame has more than one row and has numerical columns.
        You can override this method to customize the logic for generating charts.

        Args:
            df (pd.DataFrame): The DataFrame to check.

        Returns:
            bool: True if a chart should be generated, False otherwise.
        """

        if len(df) > 1 and df.select_dtypes(include=['number']).shape[1] > 0:
            return True

        return False

    def generate_rewritten_question(self, last_question: str, new_question: str, **kwargs) -> str:
        """
        **Example:**
        ```python
        rewritten_question = vn.generate_rewritten_question("Who are the top 5 customers by sales?", "Show me their email addresses")
        ```

        Generate a rewritten question by combining the last question and the new question if they are related. If the new question is self-contained and not related to the last question, return the new question.

        Args:
            last_question (str): The previous question that was asked.
            new_question (str): The new question to be combined with the last question.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The combined question if related, otherwise the new question.
        """
        if last_question is None:
            return new_question

        prompt = [
            self.system_message("Your goal is to combine a sequence of questions into a singular question if they are related. If the second question does not relate to the first question and is fully self-contained, return the second question. Return just the new combined question with no additional explanations. The question should theoretically be answerable with a single SQL statement."),
            self.user_message("First question: " + last_question + "\nSecond question: " + new_question),
        ]

        return self.submit_prompt(prompt=prompt, **kwargs)

    def generate_followup_questions(
        self, question: str, sql: str, df: pd.DataFrame, n_questions: int = 5, **kwargs
    ) -> list:
        """
        **Example:**
        ```python
        vn.generate_followup_questions("What are the top 10 customers by sales?", sql, df)
        ```

        Generate a list of followup questions that you can ask Vanna.AI.

        Args:
            question (str): The question that was asked.
            sql (str): The LLM-generated SQL query.
            df (pd.DataFrame): The results of the SQL query.
            n_questions (int): Number of follow-up questions to generate.

        Returns:
            list: A list of followup questions that you can ask Vanna.AI.
        """

        message_log = [
            self.system_message(
                f"You are a helpful data assistant. The user asked the question: '{question}'\n\nThe SQL query for this question was: {sql}\n\nThe following is a pandas DataFrame with the results of the query: \n{df.to_markdown()}\n\n"
            ),
            self.user_message(
                f"Generate a list of {n_questions} followup questions that the user might ask about this data. Respond with a list of questions, one per line. Do not answer with any explanations -- just the questions. Remember that there should be an unambiguous SQL query that can be generated from the question. Prefer questions that are answerable outside of the context of this conversation. Prefer questions that are slight modifications of the SQL query that was generated that allow digging deeper into the data. Each question will be turned into a button that the user can click to generate a new SQL query so don't use 'example' type questions. Each question must have a one-to-one correspondence with an instantiated SQL query." +
                self._response_language()
            ),
        ]

        llm_response = self.submit_prompt(message_log, **kwargs)

        numbers_removed = re.sub(r"^\d+\.\s*", "", llm_response, flags=re.MULTILINE)
        return numbers_removed.split("\n")

    def generate_questions(self, **kwargs) -> List[str]:
        """
        **Example:**
        ```python
        vn.generate_questions()
        ```

        Generate a list of questions that you can ask Vanna.AI.
        """
        question_sql = self.get_similar_question_sql(question="", **kwargs)

        return [q["question"] for q in question_sql]

    def generate_summary(self, question: str, df: pd.DataFrame, **kwargs) -> str:
        """
        **Example:**
        ```python
        vn.generate_summary("What are the top 10 customers by sales?", df)
        ```

        Generate a summary of the results of a SQL query.

        Args:
            question (str): The question that was asked.
            df (pd.DataFrame): The results of the SQL query.

        Returns:
            str: The summary of the results of the SQL query.
        """

        message_log = [
            self.system_message(
                f"You are a helpful data assistant. The user asked the question: '{question}'\n\nThe following is a pandas DataFrame with the results of the query: \n{df.to_markdown()}\n\n"
            ),
            self.user_message(
                "Briefly summarize the data based on the question that was asked. Do not respond with any additional explanation beyond the summary." +
                self._response_language()
            ),
        ]

        summary = self.submit_prompt(message_log, **kwargs)

        return summary

    # ----------------- Use Any Embeddings API ----------------- #
    @abstractmethod
    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        pass

    # ----------------- Use Any Database to Store and Retrieve Context ----------------- #
    @abstractmethod
    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        """
        This method is used to get similar questions and their corresponding SQL statements.

        Args:
            question (str): The question to get similar questions and their corresponding SQL statements for.

        Returns:
            list: A list of similar questions and their corresponding SQL statements.
        """
        pass

    @abstractmethod
    def get_related_ddl(self, question: str, **kwargs) -> list:
        """
        This method is used to get related DDL statements to a question.

        Args:
            question (str): The question to get related DDL statements for.

        Returns:
            list: A list of related DDL statements.
        """
        pass

    @abstractmethod
    def search_tables_metadata(self,
                              engine: str = None,
                              catalog: str = None,
                              schema: str = None,
                              table_name: str = None,
                              ddl: str = None,
                              size: int = 10,
                              **kwargs) -> list:
        """
        This method is used to get similar tables metadata.

        Args:
            engine (str): The database engine.
            catalog (str): The catalog.
            schema (str): The schema.
            table_name (str): The table name.
            ddl (str): The DDL statement.
            size (int): The number of tables to return.

        Returns:
            list: A list of tables metadata.
        """
        pass

    @abstractmethod
    def get_related_documentation(self, question: str, **kwargs) -> list:
        """
        This method is used to get related documentation to a question.

        Args:
            question (str): The question to get related documentation for.

        Returns:
            list: A list of related documentation.
        """
        pass

    @abstractmethod
    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        """
        This method is used to add a question and its corresponding SQL query to the training data.

        Args:
            question (str): The question to add.
            sql (str): The SQL query to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def add_ddl(self, ddl: str, **kwargs) -> str:
        """
        This method is used to add a DDL statement to the training data.

        Args:
            ddl (str): The DDL statement to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def add_documentation(self, documentation: str, **kwargs) -> str:
        """
        This method is used to add documentation to the training data.

        Args:
            documentation (str): The documentation to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def get_training_data(self, **kwargs) -> pd.DataFrame:
        """
        Example:
        ```python
        vn.get_training_data()
        ```

        This method is used to get all the training data from the retrieval layer.

        Returns:
            pd.DataFrame: The training data.
        """
        pass

    @abstractmethod
    def remove_training_data(self, id: str, **kwargs) -> bool:
        """
        Example:
        ```python
        vn.remove_training_data(id="123-ddl")
        ```

        This method is used to remove training data from the retrieval layer.

        Args:
            id (str): The ID of the training data to remove.

        Returns:
            bool: True if the training data was removed, False otherwise.
        """
        pass

    # ----------------- Use Any Language Model API ----------------- #

    @abstractmethod
    def system_message(self, message: str) -> any:
        pass

    @abstractmethod
    def user_message(self, message: str) -> any:
        pass

    @abstractmethod
    def assistant_message(self, message: str) -> any:
        pass

    def str_to_approx_token_count(self, string: str) -> int:
        return len(string) / 4

    def add_ddl_to_prompt(
        self, initial_prompt: str, ddl_list: list[str], max_tokens: int = 14000
    ) -> str:
        if len(ddl_list) > 0:
            initial_prompt += "\n===Tables \n"

            for ddl in ddl_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(ddl)
                    < max_tokens
                ):
                    initial_prompt += f"{ddl}\n\n"

        return initial_prompt

    def add_documentation_to_prompt(
        self,
        initial_prompt: str,
        documentation_list: list[str],
        max_tokens: int = 14000,
    ) -> str:
        if len(documentation_list) > 0:
            initial_prompt += "\n===Additional Context \n\n"

            for documentation in documentation_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(documentation)
                    < max_tokens
                ):
                    initial_prompt += f"{documentation}\n\n"

        return initial_prompt

    def add_sql_to_prompt(
        self, initial_prompt: str, sql_list: list[str], max_tokens: int = 14000
    ) -> str:
        if len(sql_list) > 0:
            initial_prompt += "\n===Question-SQL Pairs\n\n"

            for question in sql_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(question["sql"])
                    < max_tokens
                ):
                    initial_prompt += f"{question['question']}\n{question['sql']}\n\n"

        return initial_prompt

    def get_sql_prompt(
        self,
        initial_prompt : str,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        **kwargs,
    ):
        """
        Example:
        ```python
        vn.get_sql_prompt(
            question="What are the top 10 customers by sales?",
            question_sql_list=[{"question": "What are the top 10 customers by sales?", "sql": "SELECT * FROM customers ORDER BY sales DESC LIMIT 10"}],
            ddl_list=["CREATE TABLE customers (id INT, name TEXT, sales DECIMAL)"],
            doc_list=["The customers table contains information about customers and their sales."],
        )

        ```

        This method is used to generate a prompt for the LLM to generate SQL.

        Args:
            question (str): The question to generate SQL for.
            question_sql_list (list): A list of questions and their corresponding SQL statements.
            ddl_list (list): A list of DDL statements.
            doc_list (list): A list of documentation.

        Returns:
            any: The prompt for the LLM to generate SQL.
        """

        if initial_prompt is None:
            initial_prompt = f"""
                You are a {self.dialect} SQL Database expert. 
                Please help to generate a SQL query to answer the question. 
                The generated SQL should use single-quotes when dealing with literal value.
                Your response should ONLY be based on the given context and follow the response guidelines and format instructions. 
            """

        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        initial_prompt += (
            "===Response Guidelines \n"
            "1. If the provided context is sufficient, please generate a valid SQL query without any explanations for the question. \n"
            "2. If the provided context is almost sufficient but requires knowledge of a specific string in a particular column, please generate an intermediate SQL query to find the distinct strings in that column. Prepend the query with a comment saying intermediate_sql \n"
            "3. If the provided context is insufficient, please explain why it can't be generated. \n"
            "4. Please use the most relevant table(s). \n"
            "5. If the question has been asked and answered before, please repeat the answer exactly as it was given before. \n"
            f"6. Ensure that the output SQL is {self.dialect} SQL Database compliant and executable, and free of syntax errors. \n"
            "7. Place the generated SQL inside a Markdown sql code block. \n"
            "8. Please generate one single SQL command, DO NOT INCLUDE ANY EXPLANATION"
        )

        message_log = [self.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is None:
                print("example is None")
            else:
                if example is not None and "question" in example and "sql" in example:
                    message_log.append(self.user_message(example["question"]))
                    message_log.append(self.assistant_message(example["sql"]))

        message_log.append(self.user_message(question))

        return message_log

    def get_llm_prompt(
        self,
        question: Union[str,list],
        **kwargs,
    ):
        """
        Example:
        ```python
        vn.get_llm_prompt(
            question="What are the top 10 customers by sales?",
        )

        ```

        This method is used to generate a prompt for the LLM to generate SQL.

        Args:
            question (str): The question to generate SQL for.

        Returns:
            any: The prompt for the LLM
        """

        initial_prompt = "You are an expert AI assistant, answer me the following question:\n"
        message_log = [self.system_message(initial_prompt)]
        if isinstance(question,list):
            message_log.extend(question)
        else:
            message_log.append(self.user_message(str(question)))
        return message_log

    def get_followup_questions_prompt(
        self,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        **kwargs,
    ) -> list:
        initial_prompt = f"The user initially asked the question: '{question}': \n\n"

        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        initial_prompt = self.add_sql_to_prompt(
            initial_prompt, question_sql_list, max_tokens=self.max_tokens
        )

        message_log = [self.system_message(initial_prompt)]
        message_log.append(
            self.user_message(
                "Generate a list of followup questions that the user might ask about this data. Respond with a list of questions, one per line. Do not answer with any explanations -- just the questions."
            )
        )

        return message_log

    @abstractmethod
    def submit_prompt(self, prompt, **kwargs) -> str:
        """
        Example:
        ```python
        vn.submit_prompt(
            [
                vn.system_message("The user will give you SQL and you will try to guess what the business question this query is answering. Return just the question without any additional explanation. Do not reference the table name in the question."),
                vn.user_message("What are the top 10 customers by sales?"),
            ]
        )
        ```

        This method is used to submit a prompt to the LLM.

        Args:
            prompt (any): The prompt to submit to the LLM.

        Returns:
            str: The response from the LLM.
        """
        pass

    def generate_question(self, sql: str, **kwargs) -> str:
        response = self.submit_prompt(
            [
                self.system_message(
                    "The user will give you SQL and you will try to guess what the business question this query is answering. Return just the question without any additional explanation. Do not reference the table name in the question."
                ),
                self.user_message(sql),
            ],
            **kwargs,
        )

        return response

    def _extract_python_code(self, markdown_string: str) -> str:
        # Regex pattern to match Python code blocks
        pattern = r"```[\w\s]*python\n([\s\S]*?)```|```([\s\S]*?)```"

        # Find all matches in the markdown string
        matches = re.findall(pattern, markdown_string, re.IGNORECASE)

        # Extract the Python code from the matches
        python_code = []
        for match in matches:
            python = match[0] if match[0] else match[1]
            python_code.append(python.strip())

        if len(python_code) == 0:
            return markdown_string

        return python_code[0]

    def _sanitize_plotly_code(self, raw_plotly_code: str) -> str:
        # Remove the fig.show() statement from the plotly code
        plotly_code = raw_plotly_code.replace("fig.show()", "")

        return plotly_code

    def generate_plotly_code(
        self, question: str = None, sql: str = None, df_metadata: str = None, max_rows: int = 20, **kwargs
    ) -> str:
        if question is not None:
            system_msg = f"The following is a pandas DataFrame that contains the results of the query that answers the question the user asked: '{question}'"
        else:
            system_msg = "The following is a pandas DataFrame "

        if sql is not None:
            system_msg += f"\n\nThe DataFrame was produced using this query: {sql}\n\n"

        system_msg += f"The following is information about the resulting pandas DataFrame 'df': \n{df_metadata}"

        #  place the generated code inside a Markdown code block
        message_log = [
            self.system_message(system_msg),
            self.user_message(f"""
                Can you generate the Python plotly code to chart the results of the dataframe? 
                Assume the data is in a pandas dataframe called 'df'. 
                If the dataframe has more than {max_rows} rows, 
                use 'df.head({max_rows})' to limit the data.
                If there is only one value in the dataframe, use an Indicator. 
                Respond with only Python code
                Do not answer with any explanations -- just the code.
                """
            ),
        ]

        plotly_code = self.submit_prompt(message_log, kwargs=kwargs)

        return self._sanitize_plotly_code(self._extract_python_code(plotly_code))

    # ----------------- Connect to Any Database to run the Generated SQL ----------------- #

    def connect_to_snowflake(
        self,
        account: str,
        username: str,
        password: str,
        database: str,
        role: Union[str, None] = None,
        warehouse: Union[str, None] = None,
        **kwargs
    ):
        """ Connect to Snowflake Database
        
        If account/username/password/database startswith "my-", 
            respective value is read from environ var
        """
        try:
            snowflake = __import__("snowflake.connector")
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method, run command:"
                " \npip install vanna[snowflake]"
            )
        
        if not username or username.startswith(PREFIX_MY):
            username = os.getenv("SNOWFLAKE_USERNAME")
            if not username:
                raise ImproperlyConfigured("Please set SNOWFLAKE_USERNAME env.")

        if not password or password.startswith(PREFIX_MY):
            password = os.getenv("SNOWFLAKE_PASSWORD")
            if not password:
                raise ImproperlyConfigured("Please set SNOWFLAKE_PASSWORD env.")

        if not account or account.startswith(PREFIX_MY):
            account = os.getenv("SNOWFLAKE_ACCOUNT")
            if not account:
                raise ImproperlyConfigured("Please set SNOWFLAKE_ACCOUNT env.")

        if not database or database.startswith(PREFIX_MY):
            database = os.getenv("SNOWFLAKE_DATABASE")
            if not database:
                raise ImproperlyConfigured("Please set SNOWFLAKE_DATABASE env.")

        try:
            conn = snowflake.connector.connect(
                    user=username,
                    password=password,
                    account=account,
                    database=database,
                    client_session_keep_alive=True,
                    **kwargs)
        except Exception as e:
            raise ValidationError(f"connect_to_snowflake() failed:\n {str(e)}")

        def run_sql_snowflake(sql: str) -> pd.DataFrame:
            cs = conn.cursor()

            if role is not None:
                cs.execute(f"USE ROLE {role}")

            if warehouse is not None:
                cs.execute(f"USE WAREHOUSE {warehouse}")

            cs.execute(f"USE DATABASE {database}")
            cur = cs.execute(sql)
            results = cur.fetchall()

            # Create a pandas dataframe from the results
            df = pd.DataFrame(results, columns=[desc[0] for desc in cur.description])

            return df

        self.dialect = "Snowflake"
        self.run_sql = run_sql_snowflake
        self.run_sql_is_set = True

    def connect_to_sqlite(self, url: str, check_same_thread: bool = False,  **kwargs):
        """
        Connect to a SQLite database. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]

        Args:
            url (str): The URL of the database to connect to.
            check_same_thread (str): Allow the connection may be accessed in multiple threads.
        Returns:
            None
        """
        # Path to save the downloaded database
        if url.startswith("http") or url.startswith("file"):
            path = os.path.basename(urlparse(url).path)

            # Download the database if it doesn't exist
            if not os.path.exists(path):
                response = requests.get(url)
                response.raise_for_status()  # Check that the request was successful
                with open(path, "wb") as f:
                    f.write(response.content)
                url = path
        else:
            if not os.path.exists(url):
                raise FileNotFoundError(f"File not found: {url}")


        try:
            # Connect to the database
            conn = sqlite3.connect(
                url,
                check_same_thread=check_same_thread,
                **kwargs)
        except Exception as e:
            raise ValidationError(f"connect_to_sqlite() failed:\n {str(e)}")

        def run_sql_sqlite(sql: str):
            return pd.read_sql_query(sql, conn)

        self.dialect = "SQLite"
        self.run_sql = run_sql_sqlite
        self.run_sql_is_set = True

    def connect_to_postgres(
        self,
        host: str = None,
        dbname: str = None,
        user: str = None,
        password: str = None,
        port: int = None,
        **kwargs
    ):

        """
        Connect to postgres using the psycopg2 connector. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]
        **Example:**
        ```python
        vn.connect_to_postgres(
            host="myhost",
            dbname="mydatabase",
            user="myuser",
            password="mypassword",
            port=5432
        )
        ```
        Args:
            host (str): The postgres host.
            dbname (str): The postgres database name.
            user (str): The postgres user.
            password (str): The postgres password.
            port (int): The postgres Port.
        """

        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install vanna[postgres]"
            )

        if not host or host.startswith(PREFIX_MY):
            host = os.getenv("PG_HOST")
            if not host:
                raise ImproperlyConfigured("Please set PG_HOST env")

        if not dbname or dbname.startswith(PREFIX_MY):
            dbname = os.getenv("PG_DATABASE")
            if not dbname:
                raise ImproperlyConfigured("Please set PG_DATABASE env")

        if not user or user.startswith(PREFIX_MY):
            user = os.getenv("PG_USER")
            if not user:
                raise ImproperlyConfigured("Please set PG_USER env")

        if not password or password.startswith(PREFIX_MY):
            password = os.getenv("PG_PASSWORD")
            if not password:
                raise ImproperlyConfigured("Please set PG_PASSWORD env")

        if not port or port.startswith(PREFIX_MY):
            port = os.getenv("PG_PORT")
            if not port:
                raise ImproperlyConfigured("Please set PG_PORT env")

        try:
            conn = psycopg2.connect(
                host=host,
                dbname=dbname,
                user=user,
                password=password,
                port=port,
                **kwargs)
        except psycopg2.Error as e:
            raise ValidationError(f"connect_to_postgres() failed:\n {str(e)}")

        def psycopg2_connect():
            return psycopg2.connect(
                        host=host, 
                        dbname=dbname,
                        user=user, 
                        password=password, 
                        port=port, 
                        **kwargs)


        def run_sql_postgres(sql: str) -> Union[pd.DataFrame, None]:
            conn = None
            try:
                conn = psycopg2_connect()  # Initial connection attempt
                cs = conn.cursor()
                cs.execute(sql)
                results = cs.fetchall()

                # Create a pandas dataframe from the results
                df = pd.DataFrame(results, columns=[desc[0] for desc in cs.description])
                return df

            except psycopg2.InterfaceError as e:
                # Attempt to reconnect and retry the operation
                if conn:
                    conn.close()  # Ensure any existing connection is closed
                conn = psycopg2_connect()
                cs = conn.cursor()
                cs.execute(sql)
                results = cs.fetchall()

                # Create a pandas dataframe from the results
                df = pd.DataFrame(results, columns=[desc[0] for desc in cs.description])
                return df

            except psycopg2.Error as e:
                if conn:
                    conn.rollback()
                    raise ValidationError(e)

            except Exception as e:
                conn.rollback()
                raise e

        self.dialect = "PostgreSQL"
        self.run_sql_is_set = True
        self.run_sql = run_sql_postgres


    def connect_to_mysql(
        self,
        host: str = None,
        dbname: str = None,
        user: str = None,
        password: str = None,
        port: int = None,
        **kwargs
    ):

        try:
            import pymysql.cursors
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install PyMySQL"
            )

        if not host or host.startswith(PREFIX_MY):
            host = os.getenv("MYSQL_HOST")
            if not host:
                raise ImproperlyConfigured("Please set MYSQL_HOST")

        if not dbname or dbname.startswith(PREFIX_MY):
            dbname = os.getenv("MYSQL_DATABASE")
            if not dbname:
                raise ImproperlyConfigured("Please set MYSQL_DATABASE")

        if not user or user.startswith(PREFIX_MY):
            user = os.getenv("MYSQL_USER")
            if not user:
                raise ImproperlyConfigured("Please set MYSQL_USER")

        if not password or password.startswith(PREFIX_MY):
            password = os.getenv("MYSQL_PASSWORD")
            if not password:
                raise ImproperlyConfigured("Please set MYSQL_PASSWORD")

        if not port or port.startswith(PREFIX_MY):
            port = os.getenv("MYSQL_PORT")
            if not port:
                raise ImproperlyConfigured("Please set MYSQL_PORT")

        conn = None
        try:
            conn = pymysql.connect(
                    host=host,
                    user=user,
                    password=password,
                    database=dbname,
                    port=port,
                    cursorclass=pymysql.cursors.DictCursor,
                    **kwargs)
        except pymysql.Error as e:
            raise ValidationError(f"connect_to_mysql() failed:\n {str(e)}")

        def run_sql_mysql(sql: str) -> Union[pd.DataFrame, None]:
            if conn:
                try:
                    conn.ping(reconnect=True)
                    cs = conn.cursor()
                    cs.execute(sql)
                    results = cs.fetchall()

                    # Create a pandas dataframe from the results
                    df = pd.DataFrame(
                        results, columns=[desc[0] for desc in cs.description]
                    )
                    return df

                except pymysql.Error as e:
                    conn.rollback()
                    raise ValidationError(e)

                except Exception as e:
                    conn.rollback()
                    raise e

        self.dialect = "MySQL"
        self.run_sql_is_set = True
        self.run_sql = run_sql_mysql

    def connect_to_clickhouse(
        self,
        host: str = None,
        dbname: str = None,
        user: str = None,
        password: str = None,
        port: int = None,
        **kwargs
    ):

        try:
            import clickhouse_connect
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install clickhouse_connect"
            )

        if not host or host.startswith(PREFIX_MY):
            host = os.getenv("CLICKHOUSE_HOST")
            if not host:
                raise ImproperlyConfigured("Please set CLICKHOUSE_HOST")

        if not dbname or dbname.startswith(PREFIX_MY):
            dbname = os.getenv("CLICKHOUSE_DATABASE")
            if not dbname:
                raise ImproperlyConfigured("Please set CLICKHOUSE_DATABASE")

        if not user or user.startswith(PREFIX_MY):
            user = os.getenv("CLICKHOUSE_USER")
            if not user:
                raise ImproperlyConfigured("Please set CLICKHOUSE_USER")

        if not password or password.startswith(PREFIX_MY):
            password = os.getenv("CLICKHOUSE_PASSWORD")
            if not password:
                raise ImproperlyConfigured("Please set CLICKHOUSE_PASSWORD")

        if not port or port.startswith(PREFIX_MY):
            port = os.getenv("CLICKHOUSE_PORT")
            if not port:
                raise ImproperlyConfigured("Please set CLICKHOUSE_PORT")

        conn = None
        try:
            conn = clickhouse_connect.get_client(
                    host=host,
                    port=port,
                    username=user,
                    password=password,
                    database=dbname,
                    **kwargs)
            print(conn)
        except Exception as e:
            raise ValidationError(f"connect_to_clickhouse() failed:\n {str(e)}")

        def run_sql_clickhouse(sql: str) -> Union[pd.DataFrame, None]:
            if conn:
                try:
                    result = conn.query(sql)
                    results = result.result_rows

                    # Create a pandas dataframe from the results
                    df = pd.DataFrame(results, columns=result.column_names)
                    return df

                except Exception as e:
                    raise e

        self.dialect = "ClickHouse"
        self.run_sql_is_set = True
        self.run_sql = run_sql_clickhouse

    def connect_to_oracle(
        self,
        user: str = None,
        password: str = None,
        dsn: str = None,
        **kwargs
    ):

        """
        Connect to an Oracle db using oracledb package. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]
        **Example:**
        ```python
        vn.connect_to_oracle(
        user="username",
        password="password",
        dns="host:port/sid",
        )
        ```
        Args:
            USER (str): Oracle db user name.
            PASSWORD (str): Oracle db user password.
            DSN (str): Oracle db host ip - host:port/sid.
        """

        try:
            import oracledb
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install oracledb"
            )

        if not dsn or dsn.startswith(PREFIX_MY):
            dsn = os.getenv("ORACLE_DSN")
            if not dsn:
                raise ImproperlyConfigured("Please set ORACLE_DSN (host:port/sid)")

        if not user or user.startswith(PREFIX_MY):
            user = os.getenv("ORACLE_USER")
            if not user:
                raise ImproperlyConfigured("Please set ORACLE_USER")

        if not password or password.startswith(PREFIX_MY):
            password = os.getenv("ORACLE_PASSWORD")
            if not password:
                raise ImproperlyConfigured("Please set ORACLE_PASSWORD")

        conn = None

        try:
            conn = oracledb.connect(
                user=user,
                password=password,
                dsn=dsn,
                **kwargs)
        except oracledb.Error as e:
            raise ValidationError(f"connect_to_oracle() failed:\n {str(e)}")

        def run_sql_oracle(sql: str) -> Union[pd.DataFrame, None]:
            if conn:
                try:
                    sql = sql.rstrip()
                    if sql.endswith(';'): #fix for a known problem with Oracle db where an extra ; will cause an error.
                        sql = sql[:-1]

                    cs = conn.cursor()
                    cs.execute(sql)
                    results = cs.fetchall()

                    # Create a pandas dataframe from the results
                    df = pd.DataFrame(
                        results, columns=[desc[0] for desc in cs.description]
                    )
                    return df

                except oracledb.Error as e:
                    conn.rollback()
                    raise ValidationError(e)

                except Exception as e:
                    conn.rollback()
                    raise e

        self.dialect = "Oracle"
        self.run_sql_is_set = True
        self.run_sql = run_sql_oracle

    def connect_to_bigquery(
        self,
        cred_file_path: str = None,
        project_id: str = None,
        **kwargs
    ):
        """
        Connect to gcs using the bigquery connector. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]
        **Example:**
        ```python
        vn.connect_to_bigquery(
            project_id="myprojectid",
            cred_file_path="path/to/credentials.json",
        )
        ```
        Args:
            project_id (str): The gcs project id.
            cred_file_path (str): The gcs credential file path
        """

        try:
            from google.api_core.exceptions import GoogleAPIError
            from google.cloud import bigquery
            from google.oauth2 import service_account
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method, run command:"
                " \npip install vanna[bigquery]"
            )

        if not project_id or project_id.startswith(PREFIX_MY):
            project_id = os.getenv("GOOGLE_PROJECT_ID")
            if not project_id:
                raise ImproperlyConfigured("Please set GOOGLE_PROJECT_ID: Google Cloud Project ID.")

        import sys

        if "google.colab" in sys.modules:
            try:
                from google.colab import auth

                auth.authenticate_user()
            except Exception as e:
                raise ImproperlyConfigured(e)
        else:
            print("Not using Google Colab.")

        conn = None

        if not cred_file_path:
            try:
                conn = bigquery.Client(project=project_id)
            except:
                print("Could not found any google cloud implicit credentials")
        else:
            # Validate file path and pemissions
            validate_config_path(cred_file_path)

        if not conn:
            with open(cred_file_path, "r") as f:
                credentials = service_account.Credentials.from_service_account_info(
                    json.loads(f.read()),
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )

            try:
                conn = bigquery.Client(
                    project=project_id,
                    credentials=credentials,
                    **kwargs )
            except:
                raise ImproperlyConfigured(
                    "Could not connect to bigquery please correct credentials"
                )

        def run_sql_bigquery(sql: str) -> Union[pd.DataFrame, None]:
            if conn:
                job = conn.query(sql)
                df = job.result().to_dataframe()
                return df
            return None

        self.dialect = "BigQuery"
        self.run_sql_is_set = True
        self.run_sql = run_sql_bigquery

    def connect_to_duckdb(self, url: str, init_sql: str = None, **kwargs):
        """
        Connect to a DuckDB database. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]

        Args:
            url (str): The URL of the database to connect to. Use :memory: to create an in-memory database. Use md: or motherduck: to use the MotherDuck database.
            init_sql (str, optional): SQL to run when connecting to the database. Defaults to None.

        Returns:
            None
        """
        try:
            import duckdb
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install vanna[duckdb]"
            )
        # URL of the database to download
        if url == ":memory:" or url == "":
            path = ":memory:"
        else:
            # Path to save the downloaded database
            print(os.path.exists(url))
            if os.path.exists(url):
                path = url
            elif url.startswith("md") or url.startswith("motherduck"):
                path = url
            else:
                path = os.path.basename(urlparse(url).path)
                # Download the database if it doesn't exist
                if not os.path.exists(path):
                    response = requests.get(url)
                    response.raise_for_status()  # Check that the request was successful
                    with open(path, "wb") as f:
                        f.write(response.content)

        try:
            # Connect to the database
            conn = duckdb.connect(path, **kwargs)
            if init_sql:
                conn.query(init_sql)
        except Exception as e:
            raise ValidationError(f"connect_to_duckdb() failed:\n {str(e)}") 

        def run_sql_duckdb(sql: str):
            return conn.query(sql).to_df()

        self.dialect = "DuckDB"
        self.run_sql = run_sql_duckdb
        self.run_sql_is_set = True

    def connect_to_mssql(self, odbc_conn_str: str, **kwargs):
        """
        Connect to a Microsoft SQL Server database. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]

        Args:
            odbc_conn_str (str): The ODBC connection string.

        Returns:
            None
        """
        if not odbc_conn_str or odbc_conn_str.startswith(PREFIX_MY):
            odbc_conn_str = os.getenv("MSSQL_ODBC_CONN_STR")
            if not odbc_conn_str:
                raise ImproperlyConfigured("Please set MSSQL_ODBC_CONN_STR env.")

        try:
            import pyodbc
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: pip install pyodbc"
            )

        try:
            import sqlalchemy as sa
            from sqlalchemy.engine import URL
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: pip install sqlalchemy"
            )

        connection_url = URL.create(
            "mssql+pyodbc", query={"odbc_connect": odbc_conn_str}
        )

        from sqlalchemy import create_engine

        try:
            engine = create_engine(connection_url, **kwargs)
        except Exception as e:
            raise ValidationError(f"connect_to_mssql() failed:\n {str(e)}")

        def run_sql_mssql(sql: str):
            # Execute the SQL statement and return the result as a pandas DataFrame
            with engine.begin() as conn:
                df = pd.read_sql_query(sa.text(sql), conn)
                conn.close()
                return df

            raise Exception("Couldn't run sql")

        self.dialect = "Microsoft SQL Server"
        self.run_sql = run_sql_mssql
        self.run_sql_is_set = True

    def connect_to_presto(
        self,
        host: str,
        catalog: str = 'hive',
        schema: str = 'default',
        user: str = None,
        password: str = None,
        port: int = None,
        combined_pem_path: str = None,
        protocol: str = 'https',
        requests_kwargs: dict = None,
        **kwargs
    ):
      """
        Connect to a Presto database using the specified parameters.

        Args:
            host (str): The host address of the Presto database.
            catalog (str): The catalog to use in the Presto environment.
            schema (str): The schema to use in the Presto environment.
            user (str): The username for authentication.
            password (str): The password for authentication.
            port (int): The port number for the Presto connection.
            combined_pem_path (str): The path to the combined pem file for SSL connection.
            protocol (str): The protocol to use for the connection (default is 'https').
            requests_kwargs (dict): Additional keyword arguments for requests.

        Raises:
            DependencyError: If required dependencies are not installed.
            ImproperlyConfigured: If essential configuration settings are missing.

        Returns:
            None
      """
      try:
        from pyhive import presto
      except ImportError:
        raise DependencyError(
          "You need to install required dependencies to execute this method,"
          " run command: \npip install pyhive"
        )

      if not host or port.startswith(PREFIX_MY):
          host = os.getenv("PRESTO_HOST")
          if not host:
            raise ImproperlyConfigured("Please set PRESTO_HOST")

      if not catalog or port.startswith(PREFIX_MY):
        catalog = os.getenv("PRESTO_CATALOG")
        if not catalog:
          raise ImproperlyConfigured("Please set PRESTO_CATALOG")

      if not user or user.startswith(PREFIX_MY):
        user = os.getenv("PRESTO_USER")
        if not user:
          raise ImproperlyConfigured("Please set PRESTO_USER")

      if not password or password.startswith(PREFIX_MY):
        password = os.getenv("PRESTO_PASSWORD")
        if not password:
          raise ImproperlyConfigured("Please set PRESTO_PASSWORD")

      if not port or port.startswith(PREFIX_MY):
        port = os.getenv("PRESTO_PORT")
        if not port:
          raise ImproperlyConfigured("Please set PRESTO_PORT")

      conn = None

      try:
        if requests_kwargs is None and combined_pem_path is not None:
          # use the combined pem file to verify the SSL connection
          requests_kwargs = {
            'verify': combined_pem_path,  # 使用转换后得到的 PEM 文件进行 SSL 验证
          }
        conn = presto.Connection(host=host,
                                 username=user,
                                 password=password,
                                 catalog=catalog,
                                 schema=schema,
                                 port=port,
                                 protocol=protocol,
                                 requests_kwargs=requests_kwargs,
                                 **kwargs)
      except presto.Error as e:
        raise ValidationError(f"connect_to_presto() failed:\n {str(e)}")

      def run_sql_presto(sql: str) -> Union[pd.DataFrame, None]:
        if conn:
          try:
            sql = sql.rstrip()
            # fix for a known problem with presto db where an extra ; will cause an error.
            if sql.endswith(';'):
                sql = sql[:-1]
            cs = conn.cursor()
            cs.execute(sql)
            results = cs.fetchall()

            # Create a pandas dataframe from the results
            df = pd.DataFrame(
              results, columns=[desc[0] for desc in cs.description]
            )
            return df

          except presto.Error as e:
            print(e)
            raise ValidationError(e)

          except Exception as e:
            print(e)
            raise e

      self.dialect = "Presto"
      self.run_sql_is_set = True
      self.run_sql = run_sql_presto

    def connect_to_hive(
        self,
        host: str = None,
        dbname: str = 'default',
        user: str = None,
        password: str = None,
        port: int = None,
        auth: str = 'CUSTOM',
        **kwargs
    ):
      """
        Connect to a Hive database. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]

        Args:
            host (str): The host of the Hive database.
            dbname (str): The name of the database to connect to.
            user (str): The username to use for authentication.
            password (str): The password to use for authentication.
            port (int): The port to use for the connection.
            auth (str): The authentication method to use.

        Returns:
            None
      """

      try:
        from pyhive import hive
      except ImportError:
        raise DependencyError(
          "You need to install required dependencies to execute this method,"
          " run command: \npip install pyhive"
        )

      if not host or host.startswith(PREFIX_MY):
        host = os.getenv("HIVE_HOST")
        if not host:
          raise ImproperlyConfigured("Please set HIVE_HOST")

      if not dbname or dbname.startswith(PREFIX_MY):
        dbname = os.getenv("HIVE_DATABASE")
        if not dbname:
          raise ImproperlyConfigured("Please set HIVE_DATABASE")

      if not user or user.startswith(PREFIX_MY):
        user = os.getenv("HIVE_USER")
        if not user:
          raise ImproperlyConfigured("Please set HIVE_USER")

      if not password or password.startswith(PREFIX_MY):
        password = os.getenv("HIVE_PASSWORD")
        if not password:
          raise ImproperlyConfigured("Please set HIVE_PASSWORD")
    
      if not port or port.startswith(PREFIX_MY):
        port = os.getenv("HIVE_PORT")
        if not port:
          raise ImproperlyConfigured("Please set HIVE_PORT")

      conn = None

      try:
        conn = hive.Connection(host=host,
                               username=user,
                               password=password,
                               database=dbname,
                               port=port,
                               auth=auth)
      except hive.Error as e:
        raise ValidationError(f"connect_to_hive() failed:\n {str(e)}")

      def run_sql_hive(sql: str) -> Union[pd.DataFrame, None]:
        if conn:
          try:
            cs = conn.cursor()
            cs.execute(sql)
            results = cs.fetchall()

            # Create a pandas dataframe from the results
            df = pd.DataFrame(
              results, columns=[desc[0] for desc in cs.description]
            )
            return df

          except hive.Error as e:
            print(e)
            raise ValidationError(e)

          except Exception as e:
            print(e)
            raise e

      self.dialect = "Hive"
      self.run_sql_is_set = True
      self.run_sql = run_sql_hive

    def run_sql(self, sql: str, **kwargs) -> pd.DataFrame:
        """
        Example:
        ```python
        vn.run_sql("SELECT * FROM my_table")
        ```

        Run a SQL query on the connected database.

        Args:
            sql (str): The SQL query to run.

        Returns:
            pd.DataFrame: The results of the SQL query.
        """
        raise Exception(
            "You need to connect to a database first before running vn.connect_to_snowflake(), vn.connect_to_postgres(), similar function, or manually set vn.run_sql"
        )


    def ask_adaptive(
        self,
        question: Union[str, None] = None,
        retry_num: int = 3,
        semantic_search: bool = False, # Search Schema and skip generating SQL
        skip_chart: bool = False,   # control whether to generate Plotly code
        skip_run_sql: bool = False, # control whether to execute generated SQL
        sql_row_limit: int = 20,   # control number of rows returned: -1 for no limit
        print_prompt: bool = False,    # show prompt
        print_response: bool = False,  # show response
        print_results: bool = True,   # show results
        auto_train: bool = True,
        use_last_n_message: int = 1,
        separator: str = SEPARATOR,
        tag_id: str = "",
        sleep_sec: int = 1,
    ) -> AskResult:
        """
        Enhanced adaptive prompting by augmenting prompt with error message for LLM to self-correct

        Args:
            question (str): The question to ask.
            retry_num (int): Maximum number of retries (default=2),
            semantic_search (bool): Search schema and skip generating SQL (default=False)
            skip_chart (bool): Skip generating Plotly code when True (default=False)
            skip_run_sql (bool): Skip executing generated SQL when True (default=False)
            sql_row_limit (int): Maximum number of rows to return, -1 for no limit (default=20)
            print_prompt (bool): Print prompt, useful for debugging (default=False) 
            print_response (bool): Print LLM Response, useful for debugging (default=False) 
            print_results (bool): Show results such as generated SQL, queried dataframe, plotly chart (default=True) 
            auto_train (bool): Add valid (question,generated_sql) pair to Training dataset (default=True) 
            use_last_n_message (int): 1 (default), 0 - no history, -1 - all history, N - last number of chats
            separator (str): message tag (default=80*'='),
            tag_id (str): question tag (default=""),
            sleep_sec (int) Sleep time between retries (default=1 sec),

        Returns:
            AskResult: A named tuple of 
                - SQL query (tuple)
                - df (tuple)
                - plotly (tuple)
                - figure (tuple)
                - has_error (bool)
                
            where each tuple has 3 elements (code, ts_delta, err_msg)
        """
        with suppress_warnings_only():

            tag = f" - {tag_id}" if tag_id else ""
            vn_log(f"\n{separator}\n# QUESTION {tag}:  {question}\n")

            answer = self.ask(question=question,
                            print_results=print_results, 
                            auto_train=auto_train, 
                            visualize=(not skip_chart), 
                            allow_llm_to_see_data=(not skip_run_sql), 
                            sql_row_limit=sql_row_limit, 
                            print_prompt=print_prompt, 
                            print_response=print_response, 
                            use_last_n_message=use_last_n_message,
                            semantic_search=semantic_search)
            err_msg, has_error = collect_err_msg(answer)
            if semantic_search or ((not answer.has_error) and (not has_error)):
                return answer

            if (answer.has_error and "unknown error was encountered" in err_msg):
                # re-prompt
                answer = self.ask(question=question,
                                print_results=print_results, 
                                auto_train=auto_train, 
                                visualize=(not skip_chart), 
                                allow_llm_to_see_data=(not skip_run_sql), 
                                sql_row_limit=sql_row_limit, 
                                print_prompt=print_prompt, 
                                print_response=print_response, 
                                use_last_n_message=use_last_n_message)
                err_msg, has_error = collect_err_msg(answer)
                if not answer.has_error and (not has_error):
                    return answer

            # re-prompt
            for i_retry in range(retry_num):
                vn_log(title=LogTag.RETRY, message=f"***** {i_retry+1} *****")
                question = f"""
                    Generating SQL for this question: {question}
                    results in the following error: {err_msg} .
                    Can you try to fix the error and re-generate the SQL statement?
                """           
                answer = self.ask(question=question,
                                print_results=print_results, 
                                auto_train=auto_train, 
                                visualize=(not skip_chart), 
                                allow_llm_to_see_data=(not skip_run_sql), 
                                sql_row_limit=sql_row_limit, 
                                print_prompt=print_prompt, 
                                print_response=print_response, 
                                use_last_n_message=use_last_n_message)
                if not answer.has_error and (not has_error):
                    break

                time.sleep(sleep_sec)

            return answer

    def ask(
        self,
        question: Union[str, None] = None,
        print_results: bool = True,
        auto_train: bool = True,
        visualize: bool = True,  # if False, will not generate plotly code
        allow_llm_to_see_data: bool = True,
        sql_row_limit: int = 20,   # control number of rows returned: -1 for no limit
        print_prompt: bool = False,    # show prompt
        print_response: bool = False,  # show response
        use_last_n_message: int = 1,
        semantic_search: bool = False,
        dataset: str = "default",
    ) -> AskResult:
        """
        **Example:**
        ```python
        vn.ask("What are the top 10 customers by sales?")
        ```

        Ask Vanna.AI a question and get the SQL query that answers it.

        [u1gwg] add error msg for retry 

        Args:
            question (str): The question to ask.
            print_results (bool): Whether to print the results of the SQL query.
            auto_train (bool): Whether to automatically train Vanna.AI on the question and SQL query.
            visualize (bool): Whether to generate plotly code and display the plotly figure.
            allow_llm_to_see_data (bool): execute generated SQL
            sql_row_limit (int): Maximum number of rows to return, -1 for no limit (default=20)
            print_prompt (bool): Print prompt, useful for debugging (default=False) 
            print_response (bool): Print LLM Response, useful for debugging (default=False) 
            use_last_n_message (int): 1 (default), 0 - no history, -1 - all history, N - last number of chats
            semantic_search (bool): search schema and skip generating SQL (default=False)

        Returns:
            AskResult: A named tuple of 
                - SQL query (tuple)
                - df (tuple)
                - plotly (tuple)
                - figure (tuple)
                - has_error (bool)
        """
        result_sql = None
        result_df = None
        result_py = None
        result_fig = None
        if not question:
            # question = input("Enter a question: ")
            err_msg = f"{LogTag.ERROR_INPUT} Prompt question is missing"
            result_sql = (None, 0.0, err_msg)
            return AskResult(result_sql, None, None, None, True)

        if semantic_search:
            # answer schema-related prompt
            ts_1 = time.time()
            ctx_msg = self.summarize_context(question=question,
                                             print_prompt=print_prompt,
                                             print_response=print_response)
            ts_2 = time.time()
            ts_delta = ts_2 - ts_1
            vn_log(title=LogTag.SHOW_CXT, message=" Context summary")
            display(Code(ctx_msg, language='md'))
            result_sql = (ctx_msg, ts_delta, "")
            return AskResult(result_sql, None, None, None, False)   

        # ====================
        # Generate SQL
        # ====================
        err_msg_sql = ""
        ts_delta = 0.0
        try:
            ts_1 = time.time()
            sql = self.generate_sql(
                    question=question, 
                    allow_llm_to_see_data=allow_llm_to_see_data, 
                    print_prompt=print_prompt, 
                    print_response=print_response, 
                    use_last_n_message=use_last_n_message,
                    dataset=dataset,
                )
            
            ts_2 = time.time()
            ts_delta = ts_2 - ts_1
        except Exception as e:
            err_msg_sql = f"{LogTag.ERROR_SQL} Failed to generate SQL for prompt: {question} with the following exception: \n{str(e)}"
            print(err_msg_sql)
            result_sql = (None, ts_delta, err_msg_sql)
            return AskResult(result_sql, None, None, None, True)

        if 'intermediate_sql' in sql or 'final_sql' in sql:
            sql = sql.replace('intermediate_sql', '').replace('final_sql', '')

        if not sql.strip().lower().startswith("select") and \
            not sql.strip().lower().startswith("with"):
            err_msg_sql = f"{LogTag.ERROR_SQL} the generated SQL : {sql}\n does not starts with ('select','with')"
            result_sql = (sql, ts_delta, err_msg_sql)
            return AskResult(result_sql, None, None, None, True)

        if HAS_IPYTHON and print_results:
            try:
                vn_log(title=LogTag.SHOW_SQL, message="generated SQL statement")
                display(Code(sql, language='sql'))
                result_sql = (sql, ts_delta, None)
            except Exception as e:
                err_msg_sql = f"{LogTag.ERROR} Failed to display SQL code: {sql} with the following exception: \n{str(e)}"
                print(err_msg_sql)

        # ====================
        # Execute SQL
        # ====================
        err_msg_df = ""
        ts_delta = 0.0
        if self.run_sql_is_set is False:
            err_msg_df = f"{LogTag.ERROR} If you want to run the SQL query, connect to a database first. See here: https://vanna.ai/docs/databases.html"
            print(err_msg_df)
            result_df = (sql, ts_delta, err_msg_df)
            return AskResult(result_sql, result_df, None, None, True)

        # append limit-clause 
        sql = sql.strip()
        if sql_row_limit > 0 and "limit " not in sql.lower():
            if sql[-1] == ";":
                sql = sql[:-1]  # remove last ";" if present
            sql += f" limit {sql_row_limit}"

        try:
            ts_1 = time.time()
            df = self.run_sql(sql)
            ts_2 = time.time()
            ts_delta = ts_2 - ts_1
            result_df = (df, ts_delta, None)
        except Exception as e:
            err_msg_df = f"{LogTag.ERROR_DB} Failed to execute SQL: {sql}\n {str(e)}"
            result_df = (None, ts_delta, err_msg_df)
            return AskResult(result_sql, result_df, None, None, True)

        if HAS_IPYTHON and print_results:
            try:
                vn_log(title=LogTag.SHOW_DATA, message="queried dataframe")
                display(df)
            except Exception as e:
                print(str(e))

        if df is not None and not df.empty and len(df) > 0 and auto_train:
            self.add_question_sql(question=question, sql=sql)
        else:
            err_msg_df = f"{LogTag.ERROR_DF} Invalid dataframe"
            result_df = (df, ts_delta, err_msg_df)
            return AskResult(result_sql, result_df, None, None, True)
        
        # look for words to skip chart
        if visualize and skip_chart(question):
            visualize = False

        # Only generate plotly code if visualize is True and df has data
        if not visualize:
            return AskResult(result_sql, result_df, None, None, False)

        # ====================
        # Visualize dataframe
        # ====================
        err_msg_py = ""
        ts_delta = 0.0
        try:
            ts_1 = time.time()
            plotly_code = self.generate_plotly_code(
                question=question,
                sql=sql,
                df_metadata=f"Running df.dtypes gives:\n {df.dtypes}",
                max_rows=sql_row_limit,
            )
            ts_2 = time.time()
            ts_delta = ts_2 - ts_1

            if HAS_IPYTHON and print_results:
                vn_log(title=LogTag.SHOW_PYTHON, message="generated Plotly code")
                display(Code(plotly_code, language='python'))

            result_py = (plotly_code, ts_delta, "")
        except Exception as e:
            err_msg_py = f"{LogTag.ERROR_VIZ} Failed to generate plotly code:\n {str(e)}"
            result_py = (None, ts_delta, err_msg_py)
            return AskResult(result_sql, result_df, result_py, None, True)

        err_msg_fig = ""
        ts_delta = 0.0
        df_size = df.shape[0]
        if df_size > 0:
            try:
                ts_1 = time.time()
                max_rows = min(df_size, sql_row_limit)
                fig = self.get_plotly_figure(plotly_code=plotly_code, df=df.head(max_rows))
                ts_2 = time.time()
                ts_delta = ts_2 - ts_1

                if HAS_IPYTHON and print_results:
                    img_bytes = fig.to_image(format="png", scale=2)               
                    display(Image(img_bytes))
                    # fig.show()

                result_fig = (fig, ts_delta, "")
            except Exception as e:
                err_msg_fig = f"{LogTag.ERROR_VIZ} Failed to visualize df with plotly:\n str(e)"
                result_fig = (None, ts_delta, err_msg_fig)
                return AskResult(result_sql, result_df, result_py, result_fig, True)

        # final return
        has_error = True if any((err_msg_sql, err_msg_df, err_msg_py, err_msg_fig)) else False
        return AskResult(result_sql, result_df, result_py, result_fig, has_error)

    def ask_llm(
        self,
        question: Union[str,None],
        print_prompt: bool = True,    # show prompt
        print_response: bool = True,  # show response
    ) -> str:
        """
        Ask LLM model directly (no RAG)
        """
        print(f"[DEBUG] {question}")
        prompt = self.get_llm_prompt(question)
        vn_log(title=LogTag.SHOW_LLM, message=question, off_flag=not print_prompt)
        llm_response = self.submit_prompt(prompt=prompt, print_prompt=print_prompt, print_response=print_response)
        vn_log(title=LogTag.LLM_RESPONSE, message=llm_response, off_flag=not print_response)
        return llm_response

    def train(
        self,
        question: str = None,
        sql: str = None,
        ddl: str = None,
        engine: str = None,
        documentation: str = None,
        plan: TrainingPlan = None,
        dataset: str = "default",
    ) -> str:
        """
        **Example:**
        ```python
        vn.train()
        ```

        Train Vanna.AI on a question and its corresponding SQL query.
        If you call it with no arguments, it will check if you connected to a database and it will attempt to train on the metadata of that database.
        If you call it with the sql argument, it's equivalent to [`vn.add_question_sql()`][vanna.base.base.VannaBase.add_question_sql].
        If you call it with the ddl argument, it's equivalent to [`vn.add_ddl()`][vanna.base.base.VannaBase.add_ddl].
        If you call it with the documentation argument, it's equivalent to [`vn.add_documentation()`][vanna.base.base.VannaBase.add_documentation].
        Additionally, you can pass a [`TrainingPlan`][vanna.types.TrainingPlan] object. Get a training plan with [`vn.get_training_plan_generic()`][vanna.base.base.VannaBase.get_training_plan_generic].

        Args:
            question (str): The question to train on.
            sql (str): The SQL query to train on.
            ddl (str):  The DDL statement.
            documentation (str): The documentation to train on.
            plan (TrainingPlan): The training plan to train on.
        """
        DEBUG_FLAG=True # False #
        if ddl:
            if DEBUG_FLAG: print("\n\nAdding ddl:", ddl)
            return self.add_ddl(strip_brackets(ddl), dataset=dataset)

        if documentation:
            if DEBUG_FLAG: print("\n\nAdding documentation....")
            return self.add_documentation(documentation, dataset=dataset)

        if question and sql:
            if DEBUG_FLAG: print("\n\nAdding question/sql pair ....")
            return self.add_question_sql(question=question, sql=sql, dataset=dataset)

        if plan:
            if DEBUG_FLAG: print("\n\nAdding plan ....")
            for item in plan._plan:
                if item.item_type == TrainingPlanItem.ITEM_TYPE_DDL:
                    self.add_ddl(item.item_value, dataset=dataset)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_IS:
                    self.add_documentation(item.item_value, dataset=dataset)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_SQL:
                    self.add_question_sql(question=item.item_name, sql=item.item_value, dataset=dataset)

    def _get_databases(self) -> List[str]:
        try:
            print("Trying INFORMATION_SCHEMA.DATABASES")
            df_databases = self.run_sql("SELECT * FROM INFORMATION_SCHEMA.DATABASES")
        except Exception as e:
            print(e)
            try:
                print("Trying SHOW DATABASES")
                df_databases = self.run_sql("SHOW DATABASES")
            except Exception as e:
                print(e)
                return []

        return df_databases["DATABASE_NAME"].unique().tolist()

    def _get_information_schema_tables(self, database: str) -> pd.DataFrame:
        df_tables = self.run_sql(f"SELECT * FROM {database}.INFORMATION_SCHEMA.TABLES")

        return df_tables

    def get_training_plan_generic(self, df) -> TrainingPlan:
        """
        This method is used to generate a training plan from an information schema dataframe.

        Basically what it does is breaks up INFORMATION_SCHEMA.COLUMNS into groups of table/column descriptions that can be used to pass to the LLM.

        Args:
            df (pd.DataFrame): The dataframe to generate the training plan from.

        Returns:
            TrainingPlan: The training plan.
        """
        # For each of the following, we look at the df columns to see if there's a match:
        database_column = df.columns[
            df.columns.str.lower().str.contains("database")
            | df.columns.str.lower().str.contains("table_catalog")
        ].to_list()[0]
        schema_column = df.columns[
            df.columns.str.lower().str.contains("table_schema")
        ].to_list()[0]
        table_column = df.columns[
            df.columns.str.lower().str.contains("table_name")
        ].to_list()[0]
        columns = [database_column,
                    schema_column,
                    table_column]
        candidates = ["column_name",
                      "data_type",
                      "comment"]
        matches = df.columns.str.lower().str.contains("|".join(candidates), regex=True)
        columns += df.columns[matches].to_list()

        plan = TrainingPlan([])

        for database in df[database_column].unique().tolist():
            for schema in (
                df.query(f'{database_column} == "{database}"')[schema_column]
                .unique()
                .tolist()
            ):
                for table in (
                    df.query(
                        f'{database_column} == "{database}" and {schema_column} == "{schema}"'
                    )[table_column]
                    .unique()
                    .tolist()
                ):
                    df_columns_filtered_to_table = df.query(
                        f'{database_column} == "{database}" and {schema_column} == "{schema}" and {table_column} == "{table}"'
                    )
                    doc = f"The following columns are in the {table} table in the {database} database:\n\n"
                    doc += df_columns_filtered_to_table[columns].to_markdown()

                    plan._plan.append(
                        TrainingPlanItem(
                            item_type=TrainingPlanItem.ITEM_TYPE_IS,
                            item_group=f"{database}.{schema}",
                            item_name=table,
                            item_value=doc,
                        )
                    )

        return plan

    def get_training_plan_snowflake(
        self,
        filter_databases: Union[List[str], None] = None,
        filter_schemas: Union[List[str], None] = None,
        include_information_schema: bool = False,
        use_historical_queries: bool = True,
    ) -> TrainingPlan:
        plan = TrainingPlan([])

        if self.run_sql_is_set is False:
            raise ImproperlyConfigured("Please connect to a database first.")

        if use_historical_queries:
            try:
                print("Trying query history")
                df_history = self.run_sql(
                    """ select * from table(information_schema.query_history(result_limit => 5000)) order by start_time"""
                )

                df_history_filtered = df_history.query("ROWS_PRODUCED > 1")
                if filter_databases is not None:
                    mask = (
                        df_history_filtered["QUERY_TEXT"]
                        .str.lower()
                        .apply(
                            lambda x: any(
                                s in x for s in [s.lower() for s in filter_databases]
                            )
                        )
                    )
                    df_history_filtered = df_history_filtered[mask]

                if filter_schemas is not None:
                    mask = (
                        df_history_filtered["QUERY_TEXT"]
                        .str.lower()
                        .apply(
                            lambda x: any(
                                s in x for s in [s.lower() for s in filter_schemas]
                            )
                        )
                    )
                    df_history_filtered = df_history_filtered[mask]

                if len(df_history_filtered) > 10:
                    df_history_filtered = df_history_filtered.sample(10)

                for query in df_history_filtered["QUERY_TEXT"].unique().tolist():
                    plan._plan.append(
                        TrainingPlanItem(
                            item_type=TrainingPlanItem.ITEM_TYPE_SQL,
                            item_group="",
                            item_name=self.generate_question(query),
                            item_value=query,
                        )
                    )

            except Exception as e:
                print(e)

        databases = self._get_databases()

        for database in databases:
            if filter_databases is not None and database not in filter_databases:
                continue

            try:
                df_tables = self._get_information_schema_tables(database=database)

                print(f"Trying INFORMATION_SCHEMA.COLUMNS for {database}")
                df_columns = self.run_sql(
                    f"SELECT * FROM {database}.INFORMATION_SCHEMA.COLUMNS"
                )

                for schema in df_tables["TABLE_SCHEMA"].unique().tolist():
                    if filter_schemas is not None and schema not in filter_schemas:
                        continue

                    if (
                        not include_information_schema
                        and schema == "INFORMATION_SCHEMA"
                    ):
                        continue

                    df_columns_filtered_to_schema = df_columns.query(
                        f"TABLE_SCHEMA == '{schema}'"
                    )

                    try:
                        tables = (
                            df_columns_filtered_to_schema["TABLE_NAME"]
                            .unique()
                            .tolist()
                        )

                        for table in tables:
                            df_columns_filtered_to_table = (
                                df_columns_filtered_to_schema.query(
                                    f"TABLE_NAME == '{table}'"
                                )
                            )
                            doc = f"The following columns are in the {table} table in the {database} database:\n\n"
                            doc += df_columns_filtered_to_table[
                                [
                                    "TABLE_CATALOG",
                                    "TABLE_SCHEMA",
                                    "TABLE_NAME",
                                    "COLUMN_NAME",
                                    "DATA_TYPE",
                                    "COMMENT",
                                ]
                            ].to_markdown()

                            plan._plan.append(
                                TrainingPlanItem(
                                    item_type=TrainingPlanItem.ITEM_TYPE_IS,
                                    item_group=f"{database}.{schema}",
                                    item_name=table,
                                    item_value=doc,
                                )
                            )

                    except Exception as e:
                        print(e)
                        pass
            except Exception as e:
                print(e)

        return plan

    def get_plotly_figure(
        self, plotly_code: str, df: pd.DataFrame, dark_mode: bool = True
    ) -> plotly.graph_objs.Figure:
        """
        **Example:**
        ```python
        fig = vn.get_plotly_figure(
            plotly_code="fig = px.bar(df, x='name', y='salary')",
            df=df
        )
        fig.show()
        ```
        Get a Plotly figure from a dataframe and Plotly code.

        Args:
            df (pd.DataFrame): The dataframe to use.
            plotly_code (str): The Plotly code to use.

        Returns:
            plotly.graph_objs.Figure: The Plotly figure.
        """
        ldict = {"df": df, "px": px, "go": go}
        try:
            exec(plotly_code, globals(), ldict)

            fig = ldict.get("fig", None)
        except Exception as e:
            # Inspect data types
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            # Decision-making for plot type
            if len(numeric_cols) >= 2:
                # Use the first two numeric columns for a scatter plot
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
            elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
                # Use a bar plot if there's one numeric and one categorical column
                fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0])
            elif len(categorical_cols) >= 1 and df[categorical_cols[0]].nunique() < 10:
                # Use a pie chart for categorical data with fewer unique values
                fig = px.pie(df, names=categorical_cols[0])
            else:
                # Default to a simple line plot if above conditions are not met
                fig = px.line(df)

        if fig is None:
            return None

        if dark_mode:
            fig.update_layout(template="plotly_dark")

        return fig
