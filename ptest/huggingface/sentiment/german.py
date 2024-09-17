#!/usr/bin/env python

from germansentiment import SentimentModel


def main():
    model = SentimentModel()

    texts = [
        "Mit keinem guten Ergebniss", "Das ist gar nicht mal so gut",
        "Total awesome!", "nicht so schlecht wie erwartet",
        "Der Test verlief positiv.", "Sie fährt ein grünes Auto."]

    result = model.predict_sentiment(texts, output_probabilities=True)
    print(result)


if __name__ == '__main__':
    main()
