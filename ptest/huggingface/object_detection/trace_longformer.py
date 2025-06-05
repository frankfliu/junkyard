#!/usr/bin/env python

from transformers import pipeline


def main():
    model_id = "mrm8488/longformer-base-4096-finetuned-squadv2"
    text = "Huggingface has democratized NLP. Huge thanks to Huggingface for this."
    question = "What has Huggingface done ?"
    inputs = {"question": question, "context": text}

    pipe = pipeline("question-answering", model=model_id)

    outputs = pipe(inputs)

    print(outputs)


if __name__ == "__main__":
    main()
