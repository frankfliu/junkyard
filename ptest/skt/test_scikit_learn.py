import joblib


def main():
    model = joblib.load("model.joblib")

    inputs = [
        [
            25,
            "Private",
            226802,
            "11th",
            7,
            "Never-married",
            "Machine-op-inspct",
            "Own-child",
            "Black",
            "Male",
            0,
            0,
            40,
            "United-States",
        ]
    ]

    output = model.predict(inputs)
    print(output)


if __name__ == "__main__":
    main()
