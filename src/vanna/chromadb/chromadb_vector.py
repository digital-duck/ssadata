"""
Features
- [2024-11-23] 
    - add dataset concept support in embedding
"""
import json
from typing import List

import chromadb
import pandas as pd
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from ..base import VannaBase
from ..utils import deterministic_uuid

default_ef = embedding_functions.DefaultEmbeddingFunction()

def filter_collection_by_dataset(dataset, train_data):
    ids = []
    documents = []
    try:
        doc_list = train_data.get("documents", [])
        id_list = train_data.get("ids", [])
        # print(f"doc_list = {doc_list}")
        for n, doc in enumerate(doc_list):
            dic = json.loads(doc)
            if not dic: continue

            if dic.get("dataset", "default") == dataset:
                ids.append(id_list[n])
                documents.append(dic)
    except Exception as e:
        print(f"[ERROR] filter_collection_by_dataset():\n {str(e)}")
    return ids, documents



class ChromaDB_VectorStore(VannaBase):
    def __init__(self, config=None):
        VannaBase.__init__(self, config=config)
        if config is None:
            config = {}

        path = config.get("path", ".")
        # [DUCK]
        self.dataset = config.get("dataset", "default")
        # create collection based on dataset name
        self.embedding_function = config.get("embedding_function", default_ef)
        curr_client = config.get("client", "persistent")
        collection_metadata = config.get("collection_metadata", None) 
        # metadata={"hnsw:space": "cosine"} # l2 is the default
        self.n_results_sql = config.get("n_results_sql", config.get("n_results", 10))
        self.n_results_documentation = config.get("n_results_documentation", config.get("n_results", 10))
        self.n_results_ddl = config.get("n_results_ddl", config.get("n_results", 10))

        if curr_client == "persistent":
            self.chroma_client = chromadb.PersistentClient(
                path=path, settings=Settings(anonymized_telemetry=False)
            )
        elif curr_client == "in-memory":
            self.chroma_client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False)
            )
        elif isinstance(curr_client, chromadb.api.client.Client):
            # allow providing client directly
            self.chroma_client = curr_client
        else:
            raise ValueError(f"Unsupported client was set in config: {curr_client}")

        # [DUCK]
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.dataset,
            embedding_function=self.embedding_function,
        )

        self.documentation_collection = self.chroma_client.get_or_create_collection(
            name="documentation",
            embedding_function=self.embedding_function,
            metadata=collection_metadata,
        )
        self.ddl_collection = self.chroma_client.get_or_create_collection(
            name="ddl",
            embedding_function=self.embedding_function,
            metadata=collection_metadata,
        )
        self.sql_collection = self.chroma_client.get_or_create_collection(
            name="sql",
            embedding_function=self.embedding_function,
            metadata=collection_metadata,
        )

    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        embedding = self.embedding_function([data])
        if len(embedding) == 1:
            return embedding[0]
        return embedding


    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        """Adds a question-SQL pair to the collection with dataset metadata.

        [DUCK]

        Uses ensure_ascii=False in json.dumps() to properly handle multi-lingual content.
        This allows storing questions and SQL in any language (Chinese, Japanese, Arabic, etc.)
        without converting them to Unicode escape sequences, making the stored
        documents more readable and efficient.
        
        Args:
            question (str): The natural language question
            sql (str): The corresponding SQL query
            **kwargs: Additional arguments, supports 'dataset' key with default value "default"
        
        Returns:
            str: The generated document ID
        """
        if question is None or not question.strip() or sql is None or not sql.strip():
            return         

        dataset = kwargs.get("dataset", "default")
        doc_type = "sql"
        document = {
            "dataset": dataset,
            "question": question,
            doc_type: sql,
        }
        metadata = {
            "doc_type": doc_type,
        }

        # Serialize once and reuse
        doc_json = json.dumps(document, ensure_ascii=False)
        
        # Generate a deterministic ID
        doc_id = f"{deterministic_uuid(doc_json)}-{doc_type}"

        # Add to collection with single document
        self.collection = self.chroma_client.get_or_create_collection(
            name=dataset,
            embedding_function=self.embedding_function,
        )
        self.collection.add(
            documents=doc_json,
            embeddings=self.generate_embedding(doc_json),
            metadatas=metadata,
            ids=doc_id
        )
        
        return doc_id

    def add_question_sql_nods(self, question: str, sql: str, **kwargs) -> str:
        """Adds a question-SQL pair to the collection with dataset metadata.
        
        Uses ensure_ascii=False in json.dumps() to properly handle multi-lingual content.
        This allows storing questions and SQL in any language (Chinese, Japanese, Arabic, etc.)
        without converting them to Unicode escape sequences, making the stored
        documents more readable and efficient.
        
        Args:
            question (str): The natural language question
            sql (str): The corresponding SQL query
            **kwargs: Additional arguments, supports 'dataset' key with default value "default"
        
        Returns:
            str: The generated document ID
        """
        if question is None or not question.strip() or sql is None or not sql.strip():
            return         

        dataset = kwargs.get("dataset", "default")
        doc_type = "sql"
        document = {
            "dataset": dataset,
            "question": question,
            doc_type: sql,
        }
    
        # Serialize once and reuse
        doc_json = json.dumps(document, ensure_ascii=False)
        
        # Generate a deterministic ID
        doc_id = f"{deterministic_uuid(doc_json)}-{doc_type}"

        # Add to collection with single document

        self.sql_collection.add(
            documents=doc_json,
            embeddings=self.generate_embedding(doc_json),
            ids=doc_id
        )
        
        return doc_id

    def add_ddl(self, ddl: str, **kwargs) -> str:
        """Adds DDL script to the collection
        [DUCK]
        
        """
        if ddl is None or not ddl.strip():
            return 
                
        dataset = kwargs.get("dataset", "default")
        doc_type = "ddl"
        document = {
            "dataset": dataset,
            doc_type: ddl,
        }
        metadata = {
            "doc_type": doc_type,
        }
    
        ## Serialize once and reuse
        doc_json = json.dumps(document, ensure_ascii=False)
        
        # Generate a deterministic ID
        doc_id = f"{deterministic_uuid(doc_json)}-{doc_type}"

        # Add to collection with single document
        self.collection = self.chroma_client.get_or_create_collection(
            name=dataset,
            embedding_function=self.embedding_function,
        )
        self.collection.add(
            documents=doc_json,
            embeddings=self.generate_embedding(doc_json),
            metadatas=metadata,
            ids=doc_id
        )
        
        return doc_id        

    def add_ddl_nods(self, ddl: str, **kwargs) -> str:
        if ddl is None or not ddl.strip():
            return 
                
        doc_type = "ddl"
        document = {
            "dataset": kwargs.get("dataset", "default"),
            doc_type: ddl,
        }
    
        ## Serialize once and reuse
        doc_json = json.dumps(document, ensure_ascii=False)
        
        # Generate a deterministic ID
        doc_id = f"{deterministic_uuid(doc_json)}-{doc_type}"

        # Add to collection with single document
        self.ddl_collection.add(
            documents=doc_json,
            embeddings=self.generate_embedding(doc_json),
            ids=doc_id
        )
        
        return doc_id        

    def add_documentation(self, documentation: str, **kwargs) -> str:
        """Adds Documentation to the collection
        [DUCK]
        
        """
        if documentation is None or not documentation.strip():
            return 
        
        dataset = kwargs.get("dataset", "default")
        doc_type = "documentation"
        document = {
            "dataset": dataset,
            doc_type: documentation,
        }
        metadata = {
            "doc_type": doc_type,
        }
    
        # Serialize once and reuse
        doc_json = json.dumps(document, ensure_ascii=False)
        
        # Generate a deterministic ID
        doc_id = f"{deterministic_uuid(doc_json)}-doc"

        # Add to collection with single document
        self.collection = self.chroma_client.get_or_create_collection(
            name=dataset,
            embedding_function=self.embedding_function,
        )        
        self.collection.add(
            documents=doc_json,
            embeddings=self.generate_embedding(doc_json),
            metadatas=metadata,
            ids=doc_id
        )
        
        return doc_id   


    def add_documentation_nods(self, documentation: str, **kwargs) -> str:
        if documentation is None or not documentation.strip():
            return 
        
        doc_type = "documentation"
        document = {
            "dataset": kwargs.get("dataset", "default"),
            doc_type: documentation,
        }
    
        # Serialize once and reuse
        doc_json = json.dumps(document, ensure_ascii=False)
        
        # Generate a deterministic ID
        doc_id = f"{deterministic_uuid(doc_json)}-doc"

        # Add to collection with single document
        self.documentation_collection.add(
            documents=doc_json,
            embeddings=self.generate_embedding(doc_json),
            ids=doc_id
        )
        
        return doc_id   

    def search_tables_metadata(self,
                            engine: str = None,
                            catalog: str = None,
                            schema: str = None,
                            table_name: str = None,
                            ddl: str = None,
                            size: int = 10,
                            **kwargs) -> list:
        return []

    def get_training_data(self, **kwargs) -> pd.DataFrame:
        """Get training data from a collection
        [DUCK]
        
        """
        df = pd.DataFrame()
        dataset = kwargs.get("dataset", "default")
        # print(f"get_training_data(): dataset = {dataset}")       
        DEBUG_FLAG = False # True # 
        try:
            all_data = self.collection.get()
            if all_data is not None:
                # Extract the documents and ids
                ids, documents = filter_collection_by_dataset(dataset, all_data)
                if DEBUG_FLAG: print(f"\n documents : {documents}")

                if ids:
                    dataset_list = []
                    question_list = []
                    content_list = []
                    training_data_type_list = []
                    for doc in documents:
                        if "ddl" in doc:
                            dataset_list.append(doc.get("dataset"))
                            question_list.append(None)
                            content_list.append(doc.get("ddl"))
                            training_data_type_list.append("ddl")
                        elif "sql" in doc:
                            dataset_list.append(doc.get("dataset"))
                            question_list.append(doc.get("question"))
                            content_list.append(doc.get("sql"))
                            training_data_type_list.append("sql")
                        elif "documentation" in doc:
                            dataset_list.append(doc.get("dataset"))
                            question_list.append(None)
                            content_list.append(doc.get("documentation"))
                            training_data_type_list.append("documentation")

                    # Create a DataFrame
                    df_all = pd.DataFrame(
                        {
                            "id": ids,
                            "dataset": dataset_list,
                            "question": question_list,
                            "content": content_list,
                            "training_data_type": training_data_type_list,
                        }
                    )
                    df = pd.concat([df, df_all])
        except Exception as e:
            print(str(e))

        return df

    def get_training_data_nods(self, **kwargs) -> pd.DataFrame:
        df = pd.DataFrame()
        dataset = kwargs.get("dataset", "default")
        # print(f"get_training_data(): dataset = {dataset}")
        DEBUG_FLAG = False # True
        try:
            # get DDL metadata
            ddl_data = self.ddl_collection.get()
            if DEBUG_FLAG: print(f"\n1) ddl_data : {str(ddl_data)}")
            if ddl_data is not None:
                # Extract the documents and ids
                ids, documents = filter_collection_by_dataset(dataset, ddl_data)
                # documents = [doc for doc in ddl_data["documents"]]
                # ids = ddl_data["ids"]

                if ids:
                    # Create a DataFrame
                    df_ddl = pd.DataFrame(
                        {
                            "id": ids,
                            "dataset": [doc["dataset"] for doc in documents],
                            "question": [None for doc in documents],
                            "content": [doc["ddl"] for doc in documents],
                        }
                    )
                    df_ddl["training_data_type"] = "ddl"
                    df = pd.concat([df, df_ddl])
        except Exception as e:
            print(str(e))

        try:
            # get question/SQL pair
            sql_data = self.sql_collection.get()
            if DEBUG_FLAG: print(f"\n2) sql_data : {str(sql_data)}")
            if sql_data is not None:
                # Extract the documents and ids
                ids, documents = filter_collection_by_dataset(dataset, sql_data)
                # documents = [json.loads(doc) for doc in sql_data["documents"]]
                # ids = sql_data["ids"]

                if ids:
                    # Create a DataFrame
                    df_sql = pd.DataFrame(
                        {
                            "id": ids,
                            "dataset": [doc["dataset"] for doc in documents],
                            "question": [doc["question"] for doc in documents],
                            "content": [doc["sql"] for doc in documents],
                        }
                    )
                    df_sql["training_data_type"] = "sql"
                    df = pd.concat([df, df_sql])
        except Exception as e:
            print(str(e))

        try:
            # get bus_term metadata
            doc_data = self.documentation_collection.get()
            if DEBUG_FLAG: print(f"\n3) doc_data : {str(doc_data)}")
            if doc_data is not None:
                # Extract the documents and ids
                ids, documents = filter_collection_by_dataset(dataset, doc_data)
                # documents = [doc for doc in doc_data["documents"]]
                # ids = doc_data["ids"]

                if ids:
                    # Create a DataFrame
                    df_doc = pd.DataFrame(
                        {
                            "id": ids,
                            "dataset": [doc["dataset"] for doc in documents],
                            "question": [None for doc in documents],
                            "content": [doc["documentation"] for doc in documents],
                        }
                    )
                    df_doc["training_data_type"] = "documentation"
                    df = pd.concat([df, df_doc])
        except Exception as e:
            print(str(e))

        return df

    def remove_training_data(self, id: str, **kwargs) -> bool:
        # [DUCK]
        self.collection.delete(ids=id)
        return True

    def remove_training_data_nods(self, id: str, **kwargs) -> bool:
        if id.endswith("-sql"):
            self.sql_collection.delete(ids=id)
            return True
        elif id.endswith("-ddl"):
            self.ddl_collection.delete(ids=id)
            return True
        elif id.endswith("-doc"):
            self.documentation_collection.delete(ids=id)
            return True
        else:
            return False

    # def remove_collections(self, collection_name=None, ACCEPTED_TYPES = ["sql", "ddl", "documentation"]) -> bool:
    #     """
    #     This function is a wrapper to delete multiple collections
    #     """
    #     if not collection_name:
    #         collections = ACCEPTED_TYPES
    #     elif isinstance(collection_name, str):
    #         collections = [collection_name]
    #     elif isinstance(collection_name, list):
    #         collections = collection_name
    #     else:
    #         print(f"\t{collection_name} is unknown: Skipped")
    #         return False

    #     for c in collections:
    #         if not c in ACCEPTED_TYPES:
    #             print(f"\t{c} is unknown: Skipped")
    #             continue
                
    #         self.remove_collection(c)

    #     return True

    def remove_collection(self):
        # [DUCK]
        self.chroma_client.delete_collection(name=self.dataset)
        return True

    def remove_collection_nods(self, collection_name: str, dataset: str) -> bool:
        """
        This function can reset the collection to empty state.

        Args:
            collection_name (str): sql or ddl or documentation

        Returns:
            bool: True if collection is deleted, False otherwise
        """
        if collection_name == "sql":
            # self.chroma_client.delete_collection(name="sql")
            self.sql_collection = self.chroma_client.get_or_create_collection(
                name="sql", embedding_function=self.embedding_function
            )
            sql_data = self.sql_collection.get()
            ids, _ = filter_collection_by_dataset(dataset, sql_data)
            for collection_id in ids:
                self.remove_training_data(id=collection_id)
            return True
        elif collection_name == "ddl":
            # self.chroma_client.delete_collection(name="ddl")
            self.ddl_collection = self.chroma_client.get_or_create_collection(
                name="ddl", embedding_function=self.embedding_function
            )
            ddl_data = self.ddl_collection.get()
            ids, _ = filter_collection_by_dataset(dataset, ddl_data)
            for collection_id in ids:
                self.remove_training_data(id=collection_id)
            return True
        elif collection_name == "documentation":
            # self.chroma_client.delete_collection(name="documentation")
            self.documentation_collection = self.chroma_client.get_or_create_collection(
                name="documentation", embedding_function=self.embedding_function
            )
            doc_data = self.documentation_collection.get()
            ids, _ = filter_collection_by_dataset(dataset, doc_data)
            for collection_id in ids:
                self.remove_training_data(id=collection_id)
            return True
        else:
            return False

    @staticmethod
    def _extract_documents(query_results) -> list:
        """
        Static method to extract the documents from the results of a query.

        Args:
            query_results (pd.DataFrame): The dataframe to use.

        Returns:
            List[str] or None: The extracted documents, or an empty list or
            single document if an error occurred.
        """
        if query_results is None:
            return []

        if "documents" in query_results:
            documents = query_results["documents"]

            if len(documents) == 1 and isinstance(documents[0], list):
                try:
                    documents = [json.loads(doc) for doc in documents[0]]
                except Exception as e:
                    return documents[0]

            return documents


    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        # [DUCK]
        dataset = kwargs.get("dataset", "default")
        self.collection = self.chroma_client.get_or_create_collection(
            name=dataset,
            embedding_function=self.embedding_function,
        )
        return ChromaDB_VectorStore._extract_documents(
            self.collection.query(
                query_texts=[question],
                where={"doc_type":"sql"},
                n_results=self.n_results_sql,
            )
        )

    def get_related_ddl(self, question: str, **kwargs) -> list:
        # [DUCK]
        dataset = kwargs.get("dataset", "default")
        self.collection = self.chroma_client.get_or_create_collection(
            name=dataset,
            embedding_function=self.embedding_function,
        )
        return ChromaDB_VectorStore._extract_documents(
            self.collection.query(
                query_texts=[question],
                where={"doc_type":"ddl"},
                n_results=self.n_results_ddl,
            )
        )

    def get_related_documentation(self, question: str, **kwargs) -> list:
        # [DUCK]
        dataset = kwargs.get("dataset", "default")
        self.collection = self.chroma_client.get_or_create_collection(
            name=dataset,
            embedding_function=self.embedding_function,
        )
        return ChromaDB_VectorStore._extract_documents(
            self.collection.query(
                query_texts=[question],
                where={"doc_type":"documentation"},
                n_results=self.n_results_documentation,
            )
        )

    def get_similar_question_sql_nods(self, question: str, **kwargs) -> list:
        return ChromaDB_VectorStore._extract_documents(
            self.sql_collection.query(
                query_texts=[question],
                n_results=self.n_results_sql,
            )
        )

    def get_related_ddl_nods(self, question: str, **kwargs) -> list:
        return ChromaDB_VectorStore._extract_documents(
            self.ddl_collection.query(
                query_texts=[question],
                n_results=self.n_results_ddl,
            )
        )

    def get_related_documentation_nods(self, question: str, **kwargs) -> list:
        return ChromaDB_VectorStore._extract_documents(
            self.documentation_collection.query(
                query_texts=[question],
                n_results=self.n_results_documentation,
            )
        )
