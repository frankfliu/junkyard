from dotenv import load_dotenv


from litellm import completion


def main():
    response = completion(
        model="vertex_ai/gemini-2.5-flash",  # Specify the Vertex AI model name
        messages=[{"content": "Hello, how are you?", " role": "user"}],
    )

    print(response)


if __name__ == "__main__":
    load_dotenv()
    main()
