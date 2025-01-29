#!/usr/bin/env python

from pprint import pprint

from transformers import pipeline


def main():
    pipe = pipeline(
        "fill-mask",
        model="answerdotai/ModernBERT-base",
        # torch_dtype=torch.bfloat16,
    )

    input_text = "He walked to the [MASK]."
    results = pipe(input_text)
    pprint(results)


if __name__ == '__main__':
    main()
