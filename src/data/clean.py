from src.data.load import load_data

def audit_data():
    df = load_data()

    print("\nData Types:")
    print(df.dtypes)
    print("\nDuplicate Rows:")
    print(df.duplicated().sum())
    print("\nCardinality:")
    print(df.nunique())
    print("\nSuspicious Values:")
    print("Negative LeadTime:", (df["LeadTime"] < 0).sum())
    print("Negative ADR:", (df["ADR"] < 0).sum())
    print("Zero Guests:", (
        (df["Adults"] + df["Children"].fillna(0) + df["Babies"]) == 0
    ).sum())

    print("Negative Nights:", (
        (df["StaysInWeekendNights"] < 0) |
        (df["StaysInWeekNights"] < 0)
    ).sum())

    print("Invalid IsCanceled:", (
        ~df["IsCanceled"].isin([0, 1])
    ).sum())
    print("\nADR and LeadTime Outliers:")

    for col in ["ADR", "LeadTime"]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        print("\nColumn:", col)
        print("Minimum:", df[col].min())
        print("Maximum:", df[col].max())
        print("Lower Bound:", lower)
        print("Upper Bound:", upper)
        print("Outliers:", len(outliers))
        print("\nDuplicate Investigation:")

    print("Total Rows:", len(df))
    print("Duplicate Rows:", df.duplicated().sum())
    print("Unique Rows:", len(df.drop_duplicates()))
    
    return df

if __name__ == "__main__":
    data = audit_data()