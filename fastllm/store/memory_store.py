from typing import List, Dict
from fastllm.store.istore import ChatStorageInterface


class InMemoryChatStorage(ChatStorageInterface):
    def __init__(self) -> None:
        self.storage: Dict[str, List[dict]] = {}

    def save(self, message: dict, user_id: str = "default") -> None:
        """Save a chat message to storage."""
        if user_id not in self.storage:
            self.storage[user_id] = []

        # Append the new message to the user's message list
        self.storage[user_id].append(message)

    def get_all(self, user_id: str = "default") -> List[dict]:
        """Retrieve all messages for a specific user from storage."""
        return self.storage.get(user_id, [])
