#!/usr/bin/env python

from sentence_transformers import SentenceTransformer, util


def main():
    sentences = ["I'm happy", "I'm full of happiness"]

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Compute embedding for both lists
    embedding_1 = model.encode(sentences[0], convert_to_tensor=True)
    embedding_2 = model.encode(sentences[1], convert_to_tensor=True)

    ret = util.pytorch_cos_sim(embedding_1, embedding_2)
    print(ret)


if __name__ == '__main__':
    main()
