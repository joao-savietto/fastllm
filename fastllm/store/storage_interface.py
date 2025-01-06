from abc import ABC, abstractmethod


class ChatStorageInterface(ABC):
    @abstractmethod
    def save(self, message: dict, session_id: str = "default") -> None:
        """Save a chat message to storage."""
        pass

    # @abstractmethod
    def get_all(self, session_id: str = "default") -> list[dict]:
        """Retrieve all messages for a specific user from storage."""
        pass

    @abstractmethod
    def del_session(session_id: str = "default") -> None:
        pass

    @abstractmethod
    def del_all_sessions() -> None:
        pass
