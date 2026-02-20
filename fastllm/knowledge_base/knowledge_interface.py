from typing import Any, Dict, List


class KnowledgeBaseInterface:
    def __init__(self, path: str):
        pass

    def get_collection(self, collection_name: str):
        pass

    def insert(
        self,
        collection_name: str,
        texts: List[str],
        metadatas: List[Dict] = None,
    ):
        pass

    def query(self, collection_name: str, query: str, k: int = 5):
        pass

    def wipe(self, collection_name: str):
        pass

    def get_collection_names(self) -> List[str]:
        pass
