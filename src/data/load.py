import pandas as pd

def load_raw_data(config=None):
    df = pd.read_csv("data/raw/rawdata.csv", na_values=["null", "NULL", "Null"])
    return df


def load_data():
    return load_raw_data()


if __name__ == "__main__":
    df = load_raw_data()

    print(df.shape)
    print(df.columns.tolist())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nVerification Summary:")
    print("Total Rows:", df.shape[0])
    print("Total Columns:", df.shape[1])
    print("Total Missing Values:", df.isnull().sum().sum())