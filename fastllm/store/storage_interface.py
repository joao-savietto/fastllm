from abc import ABC, abstractmethod


class ChatStorageInterface(ABC):
    @abstractmethod
    def save(self, message: dict, user_id: str = "default") -> None:
        """Save a chat message to storage."""
        pass

    @abstractmethod
    def get_all(self, user_id: str = "default") -> list[dict]:
        """Retrieve all messages for a specific user from storage."""
        pass
