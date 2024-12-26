import pandas as pd


def main():
    df = pd.read_parquet("downloads/train-00000-of-00001.parquet")
    df.to_csv('output/openorca_1k.csv', index=False)


if __name__ == '__main__':
    main()
