"""Shared clustering features and data preparation.

K-Means, Agglomerative and GMM all use this module so they cluster exactly
the same rows and the same scaled features.
"""
from sklearn.preprocessing import StandardScaler

from src.data.load import load_raw_data
from src.preprocessing.pipeline import clean_data

# Booking profile: how early, how expensive, who, and how long.
# Same set the K-Means experiments selected ("experiment3").
CLUSTERING_FEATURES = [
    "LeadTime",
    "ADR",
    "Adults",
    "Children",
    "TotalNights",
]

# Never used for clustering, with the reason (see reports/leakage_audit.md).
EXCLUDED_COLUMNS = {
    "IsCanceled": "booking outcome; used only to describe clusters afterwards",
    "ReservationStatus": "booking outcome (Canceled / Check-Out / No-Show)",
    "ReservationStatusDate": "date of the final status, known only at the end",
    "AssignedRoomType": "decided at arrival, after the booking",
    "BookingChanges": "accumulates during the booking's life",
    "Agent": "ID code; numeric distance between IDs has no meaning",
    "Company": "ID code and mostly missing",
}

ADR_MAX = 1000  # same rule as the regression task (removes the ADR = 5400 error)


def load_clustering_data():
    """Cleaned data used by every clustering algorithm."""
    df = clean_data(load_raw_data(), task="clustering")
    df = df[df["ADR"] <= ADR_MAX].reset_index(drop=True)
    df["TotalNights"] = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
    return df


def prepare_clustering_matrix(df=None):
    """Return (scaled X, fitted scaler, df).

    Clustering has no train/test split, so the scaler is fitted on all rows;
    no target is involved, so this does not leak a label.
    """
    if df is None:
        df = load_clustering_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[CLUSTERING_FEATURES])
    return X_scaled, scaler, df 