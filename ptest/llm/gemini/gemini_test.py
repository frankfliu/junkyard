import base64
import os
from io import BytesIO

from PIL import Image
from openai import OpenAI
from pydantic import BaseModel

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


def text_generation(client: OpenAI, thinking: bool = False, stream: bool = False):
    """
    curl "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer GEMINI_API_KEY" \
    -d '{
        "model": "gemini-2.5-flash-preview-05-20",
        "reasoning_effort": "low",
        "stream": true,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain to me how AI works"}
          ]
        }'
    """

    response = client.chat.completions.create(
        model="gemini-2.5-flash-preview-05-20",
        reasoning_effort="low" if thinking else None,
        stream=stream,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain to me how AI works"},
        ],
    )
    if stream:
        for chunk in response:
            print(chunk.choices[0].delta)
    else:
        print(response.choices[0].message)


def list_models(client: OpenAI):
    models = client.models.list()
    for model in models:
        print(model.id)


def get_model(client: OpenAI):
    model = client.models.retrieve("gemini-2.5-flash-preview-05-20")
    print(model.id)


def function_call(client: OpenAI):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. Chicago, IL",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [{"role": "user", "content": "What's the weather like in Chicago today?"}]
    response = client.chat.completions.create(
        model="gemini-2.5-flash-preview-05-20", messages=messages, tools=tools, tool_choice="auto"
    )
    print(response)


def structured_output(client: OpenAI):
    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    response = client.beta.chat.completions.parse(
        model="gemini-2.5-flash-preview-05-20",
        messages=[
            {"role": "system", "content": "Extract the event information."},
            {"role": "user", "content": "John and Susan are going to an AI conference on Friday."},
        ],
        response_format=CalendarEvent,
    )
    print(response.choices[0].message.parsed)


def image_understanding(client: OpenAI):
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Getting the base64 string
    base64_image = encode_image(os.path.expanduser("~/source/samples/kitten.jpg"))

    response = client.chat.completions.create(
        model="gemini-2.5-flash-preview-05-20",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64, {base64_image}"},
                    },
                ],
            }
        ],
    )
    print(response.choices[0])


def image_generation(client: OpenAI):
    response = client.images.generate(
        model="imagen-3.0-generate-002",
        prompt="A small kitten with distinctive brown, black, and some grey striped fur, typical of a tabby pattern with eyes tightly closed",
        response_format="b64_json",
        n=1,
    )

    for image_data in response.data:
        image = Image.open(BytesIO(base64.b64decode(image_data.b64_json)))
        image.show()


def audio_understanding(client: OpenAI):
    file = os.path.expanduser("~/source/samples/test_01.wav")
    with open(file, "rb") as audio_file:
        base64_audio = base64.b64encode(audio_file.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gemini-2.5-flash-preview-05-20",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Transcribe this audio",
                    },
                    {"type": "input_audio", "input_audio": {"data": base64_audio, "format": "wav"}},
                ],
            }
        ],
    )
    print(response.choices[0].message.content)


def main():
    client = OpenAI(api_key=GEMINI_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    list_models(client)
    # get_model(client)
    # text_generation(client, thinking=True, stream=True)
    # function_call(client)
    # structured_output(client)
    # image_understanding(client)
    # image_generation(client)
    # audio_understanding(client)


if __name__ == "__main__":
    main()
