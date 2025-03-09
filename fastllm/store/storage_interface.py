from abc import ABC, abstractmethod


class ChatStorageInterface(ABC):
    @abstractmethod
    def save(self, message: dict, session_id: str = "default") -> None:
        """Save a chat message to storage."""
        pass

    @abstractmethod
    def get_all(self, session_id: str = "default") -> list[dict]:
        """Retrieve all messages for a specific user from storage."""
        pass

    @abstractmethod
    def del_session(self, session_id: str = "default") -> None:
        """Delete all messages for a specific session."""
        pass

    @abstractmethod
    def del_all_sessions(self) -> None:
        """Delete all sessions and their associated messages."""
        pass

    @abstractmethod
    def set_message(
        self, index: int, message: dict, session_id: str = "default"
    ) -> None:
        """Set a specific message at a specific index for a specific session."""  # noqa: E501
        pass

    @abstractmethod
    def get_message(self, index: int, session_id: str = "default") -> dict:
        """Retrieve a message at a specific index for a given session_id."""
        pass

    @abstractmethod
    def del_message(self, index: int, session_id: str = "default") -> None:
        """Delete a specific message at a specific index for a specific session."""  # noqa: E501
        pass
