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
    full_response = ""
    current_line = ""

    for response in agent.generate(message=user_message, session_id=session_id, stream=True):        
        if 'partial_content' in response:
            new_chunk = response['partial_content'][len(full_response):]
            full_response += new_chunk

            # Process the chunk line by line
            parts = new_chunk.split('\n')
            for part in parts[:-1]:  # handle all lines except last (which may be incomplete)
                print(part, end='\n', flush=True)  # complete line -> newline
                current_line = ""  # reset on newline
            if len(parts) > 0:  # handle the remaining partial line
                print(parts[-1], end="", flush=True)
                current_line += parts[-1]
                

    print()

if __name__ == "__main__":
    main()
