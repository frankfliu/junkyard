from huggingface_hub import InferenceClient


def main():
    client = InferenceClient(model="http://127.0.0.1:8080/invocations")
    output = client.text_generation(prompt="Write a code for snake game", details=True)
    print(output)

    # for token in client.text_generation("How do you make cheese?", max_new_tokens=12, stream=True):
    #     print(token)

    # output = client.feature_extraction("Hello world")
    # print(output)


if __name__ == '__main__':
    main()
