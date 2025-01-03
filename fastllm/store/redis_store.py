from fastllm.store.istore import ChatStorageInterface
import json
import redis


class RedisChatStorage(ChatStorageInterface):
    def __init__(self,
                 host: str = "localhost",
                 port: int = 6379,
                 db: int = 0) -> None:
        self.redis_client = redis.StrictRedis(host=host, port=port, db=db)

    def save(self, message: dict, user_id: str = "default") -> None:
        """Save a chat message to storage."""
        # Get the existing list of messages for this user
        existing_messages = self.get_all(user_id)

        # Append the new message to the list and store it back in Redis
        existing_messages.append(message)
        self.redis_client.set(user_id, json.dumps(existing_messages))

    def get_all(self, user_id: str = "default") -> list[dict]:
        """Retrieve all messages for a specific user from storage."""
        # Retrieve the stored data for this user ID and parse it as JSON
        messages_json = self.redis_client.get(user_id)

        if not messages_json:
            return []

        return json.loads(messages_json.decode("utf-8"))
