from typing import List, Dict
from fastllm.store.storage_interface import ChatStorageInterface


class InMemoryChatStorage(ChatStorageInterface):
    def __init__(self) -> None:
        self.storage: Dict[str, List[dict]] = {}

    def save(self, message: dict, session_id: str = "default") -> None:
        """Save a chat message to storage."""
        if not isinstance(message, dict):
            message = message.dict()
        if session_id not in self.storage:
            self.storage[session_id] = []

        # Append the new message to the user's message list
        self.storage[session_id].append(message)

    def get_all(self, session_id: str = "default") -> List[dict]:
        """Retrieve all messages for a specific user from storage."""
        return self.storage.get(session_id, [])

    def del_session(self, session_id: str = "default"):
        """Delete all messages of the specified session."""
        if session_id in self.storage:
            del self.storage[session_id]

    def del_all_sessions(self):
        """Clear all sessions and their corresponding messages from storage."""
        self.storage.clear()
