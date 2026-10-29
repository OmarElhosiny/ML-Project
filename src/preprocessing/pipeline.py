from src.data.load import load_raw_data


def clean_data(df):
    df = df.copy()

    df["Children"] = df["Children"].fillna(0)
    df["Country"] = df["Country"].fillna("UnKnown")

    df = df[
        (df["Adults"] + df["Children"] + df["Babies"]) > 0
    ]

    df = df.drop_duplicates()

    return df


if __name__ == "__main__":
    df = load_raw_data()

    print("Rows Before Cleaning:", len(df))

    df = clean_data(df)

    print("\nFinal Cleaning Verification:")
    print("Total Rows:", len(df))
    print("Total Missing Values:", df.isnull().sum().sum())

    zero_guests = (
        (df["Adults"] + df["Children"] + df["Babies"]) == 0
    ).sum()

    print("Zero Guests:", zero_guests)
    print("Duplicate Rows:", df.duplicated().sum())
    print("Cancellation Rate:", df["IsCanceled"].mean() * 100)