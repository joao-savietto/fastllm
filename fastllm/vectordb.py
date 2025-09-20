"""
Vector database interface using ChromaDB for semantic search capabilities.

This module provides a VectorDB class that wraps ChromaDB functionality to store,
retrieve, and query vector embeddings. It supports collection management and
semantic similarity searches.

Classes:
    VectorDB: A class for managing vector collections with ChromaDB backend.
"""

import chromadb
from chromadb.utils import embedding_functions
import uuid


class VectorDB:

    def __init__(
        self,
        embedder_model: str = "distiluse-base-multilingual-cased-v2",
        path: str = "data/",
    ):
        """Initialize the VectorDB with embedding function and client.

        Args:
            embedder_model: Name of the SentenceTransformer model to use for embeddings
            path: Path to the ChromaDB persistent database directory
        """
        self.path = path
        self.em = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedder_model
        )
        self.client = chromadb.PersistentClient(path=self.path)

    def get_collection(self, collection_name: str):
        """Get or create a collection with the given name.

        Args:
            collection_name: Name of the collection to retrieve or create

        Returns:
            ChromaDB Collection object
        """
        collection = self.client.get_or_create_collection(
            collection_name.replace(" ", "_"), embedding_function=self.em
        )
        return collection

    def insert(
        self,
        collection_name: str,
        texts: list[str],
        metadatas: list[dict] = None,
    ):
        """Insert text documents into a collection.

        Args:
            collection_name: Name of the collection to insert into
            texts: List of text documents to insert
            metadatas: Optional list of metadata dictionaries for each document

        Returns:
            None
        """
        collection = self.get_collection(collection_name.replace(" ", "_"))
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        collection.add(ids=ids, documents=texts, metadatas=metadatas)

    def query(self, collection_name: str, query: str, k: int = 5):
        """Query the vector database for similar documents.

        Args:
            collection_name: Name of the collection to query
            query: Query text to search for
            k: Number of results to return (default: 5)

        Returns:
            Query results from ChromaDB
        """
        collection = self.get_collection(collection_name.replace(" ", "_"))
        results = collection.query(query_texts=query, n_results=k)
        return results

    def wipe(self, collection_name: str):
        collection_name = collection_name.replace(" ", "_")
        collection = self.get_collection(collection_name)
        collection.delete(collection.get()["ids"])

    def get_collection_names(self):
        cols = self.client.list_collections()
        return [c.name.replace("_", " ") for c in cols]
