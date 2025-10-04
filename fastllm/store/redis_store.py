import json

import redis
from fastllm.store.storage_interface import ChatStorageInterface


class RedisChatStorage(ChatStorageInterface):
    def __init__(
        self, host: str = "localhost", port: int = 6379, db: int = 0,
        redis_client: redis.StrictRedis = None,
    ) -> None:
        if redis_client is not None:
            self.redis_client = redis_client
        else:
            self.redis_client = redis.StrictRedis(host=host, port=port, db=db)

    def save(self, message: dict, session_id: str = "default") -> None:
        """Save a chat message to storage."""
        # Get the existing list of messages for this user
        if not isinstance(message, dict):
            message = message.dict()
        existing_messages = self.get_all(session_id)

        # Append the new message to the list and store it back in Redis
        existing_messages.append(message)
        self.redis_client.set(session_id, json.dumps(existing_messages))

    def get_all(self, session_id: str = "default") -> list[dict]:
        """Retrieve all messages for a specific user from storage."""
        # Retrieve the stored data for this user ID and parse it as JSON
        messages_json = self.redis_client.get(session_id)

        if not messages_json:
            return []

        return json.loads(messages_json.decode("utf-8"))

    def del_session(self, session_id: str = "default") -> None:
        """Delete all messages of the specified session."""
        self.redis_client.delete(session_id)

    def del_all_sessions(self) -> None:
        """Clear all sessions and their corresponding messages from storage."""
        for key in self.redis_client.scan_iter("*"):
            self.redis_client.delete(key)

    def set_message(
        self, index: int, message: dict, session_id: str = "default"
    ) -> None:
        """Set a specific message at a specific index for a specific session."""
        if not isinstance(message, dict):
            message = message.dict()

        messages = self.get_all(session_id)
        if 0 <= index < len(messages):
            messages[index] = message
            self.redis_client.set(session_id, json.dumps(messages))
        else:
            raise IndexError("Index out of range")

    def get_message(self, index: int, session_id: str = "default") -> dict:
        """Retrieve a message at a specific index for a given session_id."""
        messages = self.get_all(session_id)
        if 0 <= index < len(messages):
            return messages[index]
        else:
            raise IndexError("Index out of range")

    def del_message(self, index: int, session_id: str = "default") -> None:
        """Delete a specific message at a specific index for a specific session."""
        messages = self.get_all(session_id)
        if 0 <= index < len(messages):
            del messages[index]
            self.redis_client.set(session_id, json.dumps(messages))
        else:
            raise IndexError("Index out of range")
