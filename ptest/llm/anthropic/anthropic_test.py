import os

from anthropic import AnthropicVertex

PROJECT_ID = os.environ.get("PROJECT")
REGION = os.environ.get("LOCATION", "global")
MODEL = os.environ.get("MODEL", "claude-sonnet-4-6@default")


def _print_message(message, stream: bool = False):
    if stream:
        for event in message:
            if event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    print(event.delta.text, end="", flush=True)
                elif event.delta.type == "input_json_delta":
                    print(event.delta.partial_json, end="", flush=True)
            elif event.type == "message_stop":
                print()
    else:
        print(message.content)

    print()


def text_generation(client: AnthropicVertex, stream: bool = False):
    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        stream=stream,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )
    _print_message(message, stream)


def function_call(client: AnthropicVertex, stream: bool = False):
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature, either 'celsius' or 'fahrenheit'",
                    },
                },
                "required": ["location"],
            },
        }
    ]
    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        stream=stream,
        tools=tools,
        messages=[{"role": "user", "content": "What's the weather like in Chicago today?"}],
    )
    _print_message(message, stream)


def bash(client: AnthropicVertex, stream: bool = False):
    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        stream=stream,
        tools=[{"type": "bash_20250124", "name": "bash"}],
        messages=[{"role": "user", "content": "List all Python files in the current directory."}],
    )
    _print_message(message, stream)


def computer_use(client: AnthropicVertex, stream: bool = False):
    if "opus" in MODEL or "claude-sonnet-4-6" in MODEL:
        beta = "computer-use-2025-11-24"
        tool_type = "computer_20251124"
    else:
        beta = "computer-use-2025-01-24"
        tool_type = "computer_20250124"

    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        stream=stream,
        extra_headers={"anthropic-beta": beta},
        tools=[
            {
                "type": tool_type,
                "name": "computer",
                "display_width_px": 1024,
                "display_height_px": 768,
                "display_number": 1,
            }
        ],
        messages=[{"role": "user", "content": "Save a picture of a cat to my desktop."}],
    )
    _print_message(message, stream)


def memory(client: AnthropicVertex, stream: bool = False):
    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        stream=stream,
        tools=[{"type": "memory_20250818", "name": "memory"}],
        messages=[{"role": "user", "content": "Remember that my favorite color is blue."}],
    )
    _print_message(message, stream)


def prompt_caching(client: AnthropicVertex, stream: bool = False):
    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        stream=stream,
        extra_body={"cache_control": {"type": "ephemeral", "ttl": "1h"}},
        system=[
            {
                "type": "text",
                "text": "You are a helpful assistant.",
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
        messages=[{"role": "user", "content": "What are the key terms?"}],
    )
    _print_message(message, stream)


def text_editor(client: AnthropicVertex, stream: bool = False):
    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        stream=stream,
        tools=[{"type": "text_editor_20250728", "name": "str_replace_based_edit_tool", "max_characters": 10000}],
        messages=[{"role": "user", "content": "There is a syntax error in primes.py file. Can you help me fix it?"}],
    )
    _print_message(message, stream)


def web_search(client: AnthropicVertex, stream: bool = False):
    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        stream=stream,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )
    _print_message(message, stream)


def tool_search(client: AnthropicVertex, stream: bool = False):
    message = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        stream=stream,
        tools=[
            {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
            {
                "name": "get_weather",
                "description": "Get the weather at a specific location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
                "defer_loading": True,
            },
            {
                "name": "search_files",
                "description": "Search through files in the workspace",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "file_types": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["query"],
                },
                "defer_loading": True,
            },
        ],
        messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
    )
    _print_message(message, stream)


def main():
    client = AnthropicVertex(project_id=PROJECT_ID, region=REGION)
    text_generation(client, stream=True)
    # function_call(client, stream=True)
    # bash(client, stream=True)
    # computer_use(client, stream=True)
    # memory(client, stream=True)
    # prompt_caching(client, stream=True)
    # text_editor(client, stream=True)
    # web_search(client, stream=True)
    # tool_search(client, stream=True)


if __name__ == "__main__":
    main()
