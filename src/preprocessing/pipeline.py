from pathlib import Path

from src.data.load import load_raw_data


def clean_data(df, task="shared"):
    df = df.copy()

  
    df["Children"] = df["Children"].fillna(0)
    df["Country"] = df["Country"].fillna("Unknown")

    df["Agent"] = df["Agent"].astype(str).str.strip().replace({
        "NULL": "NoAgent",
        "nan": "NoAgent"
    })

    
    guests = df["Adults"] + df["Children"] + df["Babies"]
    df = df[guests > 0]

  
    nights = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
    zero_nights = (nights == 0).sum()

    print("Zero-night bookings removed:", zero_nights)

    df = df[nights > 0]
    
    df = df.drop_duplicates().reset_index(drop=True)

    if task == "regression":
        bad_adr = (df["ADR"] <= 0) | (df["ADR"] > 1000)

        print("Invalid ADR rows removed:", bad_adr.sum())

        df = df[~bad_adr].reset_index(drop=True)

    elif task not in ("shared", "classification", "clustering"):
        raise ValueError("Invalid task")

    return df


if __name__ == "__main__":
    
    raw_df = load_raw_data()

    print("Rows before cleaning:", len(raw_df))

    
    shared_df = clean_data(raw_df, task="shared")

    regression_df = clean_data(raw_df, task="regression")

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_df.to_csv(output_dir / "cleaned_shared.csv", index=False)
    regression_df.to_csv(output_dir / "cleaned_regression.csv", index=False)

    print("\nShared Dataset:")
    print("Rows:", len(shared_df))
    print("Missing values:", shared_df.isnull().sum().sum())
    print("Duplicates:", shared_df.duplicated().sum())
    print("NULL agents:", (shared_df["Agent"].str.strip() == "NULL").sum())

    print("\nRegression Dataset:")
    print("Rows:", len(regression_df))
    print("Missing values:", regression_df.isnull().sum().sum())
    print("Duplicates:", regression_df.duplicated().sum())
    print("NULL agents:", (regression_df["Agent"].str.strip() == "NULL").sum())

    print("\nCleaning completed.")
