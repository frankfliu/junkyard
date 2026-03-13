#!/bin/bash

# basic success test case
curl "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b-maas",
    "messages": [
      {
        "role": "user",
        "content": "Say this is a test."
      }
    ]
  }'

# with all parameters
curl "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b-maas",
    "messages": [
      {
        "role": "user",
        "content": "Say this is a test."
      }
    ],
    "frequency_penalty": 0.5,
    "presence_penalty": 0.5,
    "max_completion_tokens": 100,
    "n": 1,
    "seed": 42,
    "stop": ["\n", "User:"],
    "temperature": 0.8,
    "top_p": 1.0,
    "user": "test-user"
  }'

# logprobs
curl "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b-maas",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "logprobs": true,
    "top_logprobs": 2
  }'


# json mode
curl "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b-maas",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "response_format": { "type": "json_object" }
  }'

# structured output
curl "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b-maas",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": {
                "type": "object",
                "properties": {
                    "answer": { "type": "string" }
                },
                "required": ["answer"],
                "additionalProperties": false
            },
            "strict": true
        }
    }
  }'

# function call
curl "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b-maas",
    "messages": [
      {
        "role": "user",
        "content": "What is the weather like in New York today?"
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "getWeather",
          "description": "Get the current weather in a given location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string"
              }
            },
            "required": [
              "location"
            ]
          }
        }
      }
    ],
    "tool_choice": "auto"
  }'

# streaming event
curl "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b-maas",
    "messages": [
      {
        "role": "user",
        "content": "Say this is a test."
      }
    ],
    "stream":true
  }'
