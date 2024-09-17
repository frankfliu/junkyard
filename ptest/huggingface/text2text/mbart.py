from transformers import AutoTokenizer


def main():
    # model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="ro_RO")
    tokenizer.save_pretrained("output")
    src_text = " UN Chief Says There Is No Military Solution in Syria"

    model_inputs = tokenizer(src_text, return_tensors="pt")
    print(model_inputs)


if __name__ == '__main__':
    main()
