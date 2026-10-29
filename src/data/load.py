import pandas as pd

def load_data():
    df = pd.read_csv("data/raw/rawdata.csv", na_values=["null", "NULL", "Null"])

    print(df.shape)
    print(df.columns.tolist())
    print("\nMissing Valuues:")
    print(df.isnull().sum())
    print("\nVerification Summary:")
    print("Total Rows:", df.shape[0])
    print("Total Columns:", df.shape[1])
    print("Total Missing Values:", df.isnull().sum().sum())
    
    return df


if __name__ == "__main__":
    data = load_data()