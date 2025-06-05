#!/usr/bin/env python

from transformers import AutoTokenizer


def test(inputs, tokenizer, truncation=False, padding=False, max_length=None, pad_to_multiple_of=None):
    outputs = tokenizer.batch_encode_plus(
        inputs, truncation=truncation, padding=padding, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of
    )

    input_ids1 = outputs["input_ids"][0]
    input_ids2 = outputs["input_ids"][1]
    print(f"truncation={truncation}, padding={padding}, max_length={max_length}: {len(input_ids1)} / {len(input_ids2)}")


def main():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer.model_max_length = 20

    input1 = ["my question1", "context1"]
    input2 = ["question2", "context2 in the batch 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"]
    inputs = [input1, input2]

    # no truncation
    test(inputs, tokenizer, truncation=False, padding=False)
    test(inputs, tokenizer, truncation=False, padding=True)
    test(inputs, tokenizer, truncation=False, padding=True, pad_to_multiple_of=3)
    # max_length is ignored
    test(inputs, tokenizer, truncation=False, padding=True, max_length=3)
    # padding to model_max_length (20)
    test(inputs, tokenizer, truncation=False, padding="max_length")
    test(inputs, tokenizer, truncation=False, padding="max_length", pad_to_multiple_of=3)
    test(inputs, tokenizer, truncation=False, padding="max_length", max_length=0)
    test(inputs, tokenizer, truncation=False, padding="max_length", max_length=22)
    test(inputs, tokenizer, truncation=False, padding="max_length", max_length=22, pad_to_multiple_of=3)

    # truncation=True, no max_length
    test(inputs, tokenizer, truncation=True, padding=False)
    test(inputs, tokenizer, truncation="longest_first", padding=False)
    # only first will fail: Sequence to truncate too short to respect the provided max_length
    # test(inputs, tokenizer, truncation="only_first", padding=False)
    test(inputs, tokenizer, truncation="only_second", padding=False)
    test(inputs, tokenizer, truncation=True, padding=True)
    # pad_to_multiple_of is ignored
    test(inputs, tokenizer, truncation=True, padding=True, pad_to_multiple_of=3)
    # padding to model_max_length (512)
    test(inputs, tokenizer, truncation=True, padding="max_length")
    # expect to fail
    test(inputs, tokenizer, truncation=True, padding="max_length", pad_to_multiple_of=3)
    test(inputs, tokenizer, truncation=True, padding="max_length", pad_to_multiple_of=2)

    # truncation with max_length
    test(inputs, tokenizer, truncation=True, padding=False, max_length=22)
    test(inputs, tokenizer, truncation=True, padding=True, max_length=22)
    # pad_to_multiple_of is ignored
    test(inputs, tokenizer, truncation=True, padding=True, max_length=22, pad_to_multiple_of=3)
    # padding to model_max_length (512)
    test(inputs, tokenizer, truncation=True, padding="max_length", max_length=22)
    # expect to fail
    # test(inputs, tokenizer, truncation=True, padding="max_length", max_length=22, pad_to_multiple_of=3)
    test(inputs, tokenizer, truncation=True, padding="max_length", max_length=22, pad_to_multiple_of=2)


if __name__ == "__main__":
    main()
