from typing import Dict, List

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

    def del_session(self, session_id: str = "default") -> None:
        """Delete all messages of the specified session."""
        if session_id in self.storage:
            del self.storage[session_id]

    def del_all_sessions(self) -> None:
        """Clear all sessions and their corresponding messages from storage."""
        self.storage.clear()

    def set_message(
        self, index: int, message: dict, session_id: str = "default"
    ) -> None:
        """Set a specific message at a specific index for a specific session."""  # noqa: E501
        if session_id not in self.storage:
            self.storage[session_id] = []

        if 0 <= index < len(self.storage[session_id]):
            self.storage[session_id][index] = message
        else:
            raise IndexError("Index out of range")

    def get_message(self, index: int, session_id: str = "default") -> dict:
        """Retrieve a message at a specific index for a given session_id."""
        if session_id not in self.storage:
            raise KeyError(f"Session {session_id} does not exist")

        if 0 <= index < len(self.storage[session_id]):
            return self.storage[session_id][index]
        else:
            raise IndexError("Index out of range")

    def del_message(self, index: int, session_id: str = "default") -> None:
        """Delete a specific message at a specific index for a specific session."""  # noqa: E501
        if session_id not in self.storage:
            raise KeyError(f"Session {session_id} does not exist")

        if 0 <= index < len(self.storage[session_id]):
            del self.storage[session_id][index]
        else:
            raise IndexError("Index out of range")
