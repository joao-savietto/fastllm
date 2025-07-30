from fastllm.agent import Agent



def main():
    api_key = "ollama"
    system_prompt = "You are a helpful assistant."

    agent = Agent(
        model="qwen2.5-coder-32b-instruct-128k",
        base_url="http://localhost:1234/v1",
        api_key=api_key,
        system_prompt=system_prompt
    )

    session_id = "example_session"
    user_message = "Hello, how are you?"

    print(f"User: {user_message}")

    for chunk in agent.generate(message=user_message, session_id=session_id, stream=True):        
        if 'partial_content' in chunk:
            new_chunk = chunk['partial_content']
            print(new_chunk, end='', flush=True)
    print()


if __name__ == "__main__":
    main()
