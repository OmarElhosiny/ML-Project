from src.data.load import load_data

def clean_data(remove_duplicates=True):
    df = load_data()

    df["Children"] = df["Children"].fillna(0)

    print("Missing Children:", df["Children"].isnull().sum())
    df["Country"] = df["Country"].fillna("UnKnown")
    print("Missing Country:", df["Country"].isnull().sum())
    print("Rows Before:", len(df))

    df = df[
        (df["Adults"] + df["Children"] + df["Babies"]) > 0
    ]

    print("Rows After:", len(df))
    print("\nDuplicates After Cleaning:")
    print("Duplicate Rows:", df.duplicated().sum())
    print("Unique Rows:", len(df.drop_duplicates()))

    def count_duplicates(x):
     return x.duplicated().sum()
    print(df.groupby("IsCanceled").apply(count_duplicates))

    print("\nCancellation Rate Before Removing Duplicates:")
    print(df["IsCanceled"].mean() * 100)

    print("\nCancellation Rate After Removing Duplicates:")
    print(df.drop_duplicates()["IsCanceled"].mean() * 100)

    if remove_duplicates:
       df = df.drop_duplicates()

    print("\nFinal Cleaning Verification:")
    print("Total Rows:", len(df))
    print("Total Missing Values:", df.isnull().sum().sum())

    zero_guests = (
        df["Adults"] + df["Children"] + df["Babies"] == 0
    ).sum()

    print("Zero Guests:", zero_guests)
    print("Duplicate Rows:", df.duplicated().sum())
    return df


if __name__ == "__main__":
    data = clean_data()