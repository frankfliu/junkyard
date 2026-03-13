import json
import logging

from openai import APIError, OpenAI

logger = logging.getLogger(__name__)


def _send_request(client: OpenAI, kwargs: dict):
    """Sends a request and returns if it was supported."""
    try:
        client.chat.completions.create(**kwargs)
        return True, "Supported"
    except APIError as e:
        error_msg = str(e).lower()
        if any(
            keyword in error_msg
            for keyword in [
                "extra fields",
                "unsupported",
                "unexpected argument",
                "unknown parameter",
            ]
        ):
            return False, f"Not Supported: {e}"
        return False, f"Error: {e}"
    except Exception as e:
        logger.exception("send request failed")
        return False, f"Unexpected Error: {e}"


def _test_parameter(client: OpenAI, model: str, param_name, param_value):
    """Sends a request with a specific parameter and returns if it was supported."""
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Map internal test labels to actual OpenAI parameters
    if param_name == "response_format_json_schema":
        kwargs["response_format"] = param_value
    elif param_name == "web_search_tool":
        kwargs["tools"] = param_value
        kwargs["tool_choice"] = {
            "type": "function",
            "function": {"name": "web_search"},
        }
    else:
        kwargs[param_name] = param_value

    return _send_request(client, kwargs)


def test_chat_completion_parameters(client: OpenAI, model: str, results: list):
    parameters_to_test = [
        ("audio", {"voice": "alloy", "format": "wav"}),
        ("frequency_penalty", 0.5),
        ("logit_bias", {50256: -100}),
        ("logprobs", True),
        ("max_completion_tokens", 10),
        ("metadata", {"test": "value"}),
        ("modalities", ["text"]),
        ("n", 1),
        ("parallel_tool_calls", True),
        ("prediction", {"type": "content", "content": "Hello"}),
        ("presence_penalty", 0.5),
        ("reasoning_effort", "medium"),
        (
            "response_format_json_schema",
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "test_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                },
            },
        ),
        ("response_format", {"type": "json_object"}),
        ("seed", 42),
        ("service_tier", "auto"),
        ("stop", ["\n"]),
        ("store", True),
        ("stream_options", {"include_usage": True}),
        ("stream", False),
        ("temperature", 0.7),
        (
            "tools",
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    },
                }
            ],
        ),
        ("tool_choice", "auto"),
        ("top_logprobs", 2),
        ("top_p", 0.9),
        ("user", "test-user"),
        (
            "web_search_tool",
            [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                    },
                }
            ],
        ),
    ]

    for param, value in parameters_to_test:
        supported, note = _test_parameter(client, model, param, value)
        status = "PASS" if supported else "FAIL"
        logger.info(f"Testing parameter: {param}... {status}")
        results.append(
            {
                "parameter": param,
                "value": json.dumps(value),
                "status": status,
                "note": note,
            }
        )
