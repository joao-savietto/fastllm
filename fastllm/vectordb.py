import chromadb
from chromadb.utils import embedding_functions
import uuid


class VectorDB:

    def __init__(
        self,
        embedder_model: str = "distiluse-base-multilingual-cased-v2",
        path: str = "data/",
    ):
        self.path = path
        self.em = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedder_model
        )
        self.client = chromadb.PersistentClient(path=self.path)

    def get_collection(self, collection_name: str):
        collection = self.client.get_or_create_collection(
            collection_name.replace(" ", "_"), embedding_function=self.em
        )
        return collection

    def insert(
        self, collection_name: str, texts: list[str], metadatas: list[dict] = None
    ):
        collection = self.get_collection(collection_name.replace(" ", "_"))
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        collection.add(ids=ids, documents=texts, metadatas=metadatas)

    def query(self, collection_name: str, query: str, k: int = 5):
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
