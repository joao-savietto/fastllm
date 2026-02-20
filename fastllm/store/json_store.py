import json
import os
from typing import List, Dict
from fastllm.store.storage_interface import ChatStorageInterface


class JSONChatStorage(ChatStorageInterface):
    def __init__(self, storage_dir: str = "storage") -> None:
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def _get_file_path(self, session_id: str) -> str:
        # Sanitize session_id to avoid path traversal
        safe_session_id = "".join(
            c for c in session_id if c.isalnum() or c in ("-", "_")
        )
        if not safe_session_id:
            safe_session_id = "default"
        return os.path.join(self.storage_dir, f"{safe_session_id}.json")

    def _load_messages(self, session_id: str) -> List[dict]:
        file_path = self._get_file_path(session_id)
        if not os.path.exists(file_path):
            return []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def _save_messages(self, session_id: str, messages: List[dict]) -> None:
        file_path = self._get_file_path(session_id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)

    def save(self, message: dict, session_id: str = "default") -> None:
        """Save a chat message to storage."""
        if not isinstance(message, dict):
            # If message is a Pydantic model or similar
            if hasattr(message, "dict"):
                message = message.dict()
            else:
                # Fallback if it's not a dict and doesn't have .dict()
                # Although the interface suggests dict, robust handling is good
                pass

        messages = self._load_messages(session_id)
        messages.append(message)
        self._save_messages(session_id, messages)

    def get_all(self, session_id: str = "default") -> List[dict]:
        """Retrieve all messages for a specific user from storage."""
        return self._load_messages(session_id)

    def del_session(self, session_id: str = "default") -> None:
        """Delete all messages of the specified session."""
        file_path = self._get_file_path(session_id)
        if os.path.exists(file_path):
            os.remove(file_path)

    def del_all_sessions(self) -> None:
        """Clear all sessions and their corresponding messages from storage."""
        if os.path.exists(self.storage_dir):
            for filename in os.listdir(self.storage_dir):
                if filename.endswith(".json"):
                    os.remove(os.path.join(self.storage_dir, filename))

    def set_message(
        self, index: int, message: dict, session_id: str = "default"
    ) -> None:
        """Set a specific message at a specific index for a specific session."""
        if not isinstance(message, dict):
            if hasattr(message, "dict"):
                message = message.dict()

        messages = self._load_messages(session_id)

        # Even if session didn't exist (messages=[]), we check index range
        if 0 <= index < len(messages):
            messages[index] = message
            self._save_messages(session_id, messages)
        else:
            raise IndexError("Index out of range")

    def get_message(self, index: int, session_id: str = "default") -> dict:
        """Retrieve a message at a specific index for a given session_id."""
        file_path = self._get_file_path(session_id)
        if not os.path.exists(file_path):
            raise KeyError(f"Session {session_id} does not exist")

        messages = self._load_messages(session_id)

        if 0 <= index < len(messages):
            return messages[index]
        else:
            raise IndexError("Index out of range")

    def del_message(self, index: int, session_id: str = "default") -> None:
        """Delete a specific message at a specific index for a specific session."""
        file_path = self._get_file_path(session_id)
        if not os.path.exists(file_path):
            raise KeyError(f"Session {session_id} does not exist")

        messages = self._load_messages(session_id)

        if 0 <= index < len(messages):
            del messages[index]
            self._save_messages(session_id, messages)
        else:
            raise IndexError("Index out of range")
