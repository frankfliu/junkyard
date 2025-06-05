#!/usr/bin/env python
#
# pip install --pre onnxruntime-genai onnxruntime
# huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* --local-dir .

import onnxruntime_genai as og


def main():
    model = og.Model("/Users/frankfliu/source/models/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4")
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()

    # Set the max length to something sensible by default,
    # since otherwise it will be set to the entire context length
    search_options = {"max_length": 384, "batch_size": 1}
    chat_template = "<|user|>\n{input} <|end|>\n<|assistant|>"

    text = "What's 6 times 7?"
    prompt = f"{chat_template.format(input=text)}"

    input_tokens = tokenizer.encode(prompt)

    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)
    generator = og.Generator(model, params)

    print("Output: ", end="", flush=True)

    try:
        generator.append_tokens(input_tokens)
        while not generator.is_done():
            generator.generate_next_token()

            new_token = generator.get_next_tokens()[0]
            print(tokenizer_stream.decode(new_token), end="", flush=True)
    except KeyboardInterrupt:
        print("  --control+c pressed, aborting generation--")

    print()
    del generator


if __name__ == "__main__":
    main()
