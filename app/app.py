"""Streamlit application for the City Hotel (H2) ML project.

Run from the project root:
    streamlit run app/app.py

The app loads saved artifacts and saved result tables. It never retrains a
model when it starts; training only happens when the user presses the button
on the "Models" page.
"""
import subprocess
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_selection import SET_A_FEATURES  # noqa: E402
from src.inference.predict import load_model, prepare_booking  # noqa: E402

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "rawdata.csv"

MODEL_NAMES = {
    "xgb": "XGBoost",
    "rf": "Random Forest",
    "logistic": "Logistic Regression",
    "linear": "Linear Regression",
    "mlp": "MLP",
    "final": "Final model",
}

st.set_page_config(page_title="City Hotel ML", layout="wide")


# ======================================================================
# Helpers
# ======================================================================

def nice_model_name(prefix):
    return MODEL_NAMES.get(prefix, prefix.replace("_", " ").title())


def read_table(name):
    path = TABLES_DIR / name
    return pd.read_csv(path) if path.exists() else None


def show_figure(name, caption=None):
    path = FIGURES_DIR / name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"Figure not found: outputs/figures/{name}")


def saved_artifacts():
    if not ARTIFACTS_DIR.exists():
        return []
    return sorted(p.stem for p in ARTIFACTS_DIR.glob("*.joblib"))


def baseline_tables(task):
    """Main results table of each model for one task (feature-set experiments excluded)."""
    tables = []
    for path in sorted(TABLES_DIR.glob(f"*_{task}_baseline.csv")):
        table = pd.read_csv(path)
        table.insert(0, "source", path.stem.split("_")[0])
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else None


def confusion_matrices():
    """Main confusion matrix of each classification model."""
    matrices = {}
    for path in sorted(TABLES_DIR.glob("*_classification_confusion_matrix.csv")):
        prefix = path.stem.split("_")[0]
        matrices[nice_model_name(prefix)] = pd.read_csv(path, index_col=0)
    return matrices


@st.cache_data(show_spinner="Loading dataset...")
def load_raw():
    return pd.read_csv(RAW_DATA_PATH)


@st.cache_data(show_spinner="Preparing form options...")
def load_form_data():
    """Cleaned data, used only to fill the prediction form with valid values."""
    from src.data.load import load_raw_data
    from src.preprocessing.pipeline import clean_data

    return clean_data(load_raw_data(), task="regression")


@st.cache_resource(show_spinner="Loading model...")
def cached_model(name):
    return load_model(name)


# ======================================================================
# Pages
# ======================================================================

def page_overview():
    st.title(" City Hotel (H2) — Machine Learning System")
    st.write(
        "Predict the average daily rate (**ADR**), predict whether a booking will be "
        "**canceled**, and **segment** bookings into customer groups."
    )

    if not RAW_DATA_PATH.exists():
        st.error("Dataset not found. Place it at `data/raw/rawdata.csv`.")
        return

    df = load_raw()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bookings", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Duplicate rows", f"{df.duplicated().sum():,}")
    c4.metric("Cancellation rate", f"{df['IsCanceled'].mean():.1%}")

    with st.expander("First rows of the raw data"):
        st.dataframe(df.head(20))

    with st.expander("Column types and missing values"):
        info = pd.DataFrame({
            "type": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "missing_%": (df.isna().mean() * 100).round(2),
            "unique": df.nunique(),
        })
        st.dataframe(info)

    st.subheader("Exploratory analysis")
    tab_adr, tab_cancel, tab_other = st.tabs(["ADR", "Cancellation", "Other"])
    with tab_adr:
        col1, col2 = st.columns(2)
        with col1:
            show_figure("adr_distribution_zoomed.png", "ADR distribution")
            show_figure("adr_by_room_type.png", "ADR by room type")
        with col2:
            show_figure("adr_by_month.png", "ADR by month")
            show_figure("adr_by_guest_count.png", "ADR by number of guests")
    with tab_cancel:
        col1, col2 = st.columns(2)
        with col1:
            show_figure("booking_cancellation.png", "Canceled vs not canceled")
            show_figure("cancellation_by_deposit.png", "By deposit type")
        with col2:
            show_figure("cancellation_by_leadtime.png", "By lead time")
            show_figure("cancellation_by_market.png", "By market segment")
    with tab_other:
        col1, col2 = st.columns(2)
        with col1:
            show_figure("leadtime_distribution.png", "Lead time")
        with col2:
            show_figure("total_nights_distribution.png", "Total nights")


def page_models():
    st.title(" Models")
    st.write(
        "Saved models are loaded from `artifacts/`. Training is only needed once, "
        "or after the code changes."
    )

    artifacts = saved_artifacts()
    if artifacts:
        st.success(f"{len(artifacts)} saved artifacts found.")
        st.dataframe(pd.DataFrame({"artifact": artifacts}), hide_index=True)
    else:
        st.warning("No saved models yet. Train them below.")

    st.subheader("Train models")
    task = st.selectbox("Task", ["all", "regression", "classification"])
    if st.button("Train and save models", type="primary"):
        with st.spinner("Training... this can take a few minutes."):
            result = subprocess.run(
                [sys.executable, "main.py", "train", "--task", task],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
        if result.returncode == 0:
            st.success("Training finished. Models saved to artifacts/.")
            cached_model.clear()
        else:
            st.error("Training failed.")
        st.code(result.stdout[-3000:] + result.stderr[-3000:])


def page_comparison():
    st.title(" Model comparison")
    st.caption("All results are on the validation set. The test set is kept for the final evaluation.")

    st.subheader("Regression — ADR")
    regression = baseline_tables("regression")
    if regression is None:
        st.info("No regression results yet.")
    else:
        cols = [c for c in ["model", "configuration", "feature_set", "MAE", "RMSE", "R2"] if c in regression.columns]
        st.dataframe(regression[cols].round(4), hide_index=True)
        metric_cols = [c for c in ["MAE", "RMSE"] if c in regression.columns]
        if metric_cols:
            chart = regression.set_index(regression["source"].map(nice_model_name))[metric_cols]
            st.bar_chart(chart, stack=False)

    st.subheader("Classification — IsCanceled")
    classification = baseline_tables("classification")
    if classification is None:
        st.info("No classification results yet.")
    else:
        cols = [c for c in ["model", "configuration", "feature_set", "Accuracy", "Precision",
                            "Recall", "F1", "ROC_AUC"] if c in classification.columns]
        st.dataframe(classification[cols].round(4), hide_index=True)
        metric_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
                       if c in classification.columns]
        chart = classification.set_index(classification["source"].map(nice_model_name))[metric_cols]
        st.bar_chart(chart.T, stack=False)

    st.subheader("Confusion matrices")
    matrices = confusion_matrices()
    if not matrices:
        st.info("No confusion matrices yet.")
    else:
        fig, axes = plt.subplots(1, len(matrices), figsize=(4.5 * len(matrices), 4), squeeze=False)
        for ax, (name, matrix) in zip(axes[0], matrices.items()):
            values = matrix.values
            ax.imshow(values, cmap="Blues")
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    color = "white" if values[i, j] > values.max() / 2 else "black"
                    ax.text(j, i, f"{values[i, j]:,}", ha="center", va="center", color=color, fontsize=12)
            ax.set_xticks([0, 1], ["Not canceled", "Canceled"])
            ax.set_yticks([0, 1], ["Not canceled", "Canceled"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(name)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def page_tuning():
    st.title(" Hyperparameter tuning")
    tuning_files = sorted(TABLES_DIR.glob("*tuning*.csv")) + sorted(TABLES_DIR.glob("*_baseline_vs_optimized.csv"))
    if not tuning_files:
        st.info("No tuning results yet.")
        return
    for path in tuning_files:
        with st.expander(path.stem.replace("_", " ").capitalize(), expanded=False):
            st.dataframe(pd.read_csv(path), hide_index=True)


def booking_form(key):
    """Form with the 21 Set A features. Returns a booking dict or None."""
    data = load_form_data()

    def options(column):
        return data[column].value_counts().index.tolist()

    def clean_label(value):
        return str(value).strip()

    with st.form(key):
        st.markdown("**Stay**")
        c1, c2, c3 = st.columns(3)
        arrival = c1.date_input(
            "Arrival date",
            value=date(2016, 8, 15),
            min_value=date(2015, 7, 1),
            max_value=date(2017, 8, 31),
            help="The dataset covers July 2015 to August 2017.",
        )
        lead_time = c2.number_input("Lead time (days before arrival)", 0, 700, 60)
        weekend_nights = c3.number_input("Weekend nights", 0, 20, 1)
        c1, c2, c3 = st.columns(3)
        week_nights = c1.number_input("Week nights", 0, 30, 2)
        meal = c2.selectbox("Meal", options("Meal"), format_func=clean_label)
        room = c3.selectbox("Reserved room type", options("ReservedRoomType"), format_func=clean_label)

        st.markdown("**Guests**")
        c1, c2, c3 = st.columns(3)
        adults = c1.number_input("Adults", 0, 4, 2)
        children = c2.number_input("Children", 0, 3, 0)
        babies = c3.number_input("Babies", 0, 2, 0)

        st.markdown("**Booking source**")
        c1, c2, c3 = st.columns(3)
        market = c1.selectbox("Market segment", options("MarketSegment"))
        channel = c2.selectbox("Distribution channel", options("DistributionChannel"))
        customer = c3.selectbox("Customer type", options("CustomerType"))
        c1, c2, c3 = st.columns(3)
        deposit = c1.selectbox("Deposit type", options("DepositType"), format_func=clean_label)
        country = c2.selectbox("Country", options("Country"))
        agent = c3.selectbox("Agent", options("Agent"))

        st.markdown("**Guest history**")
        c1, c2, c3 = st.columns(3)
        repeated = c1.checkbox("Repeated guest")
        prev_cancel = c2.number_input("Previous cancellations", 0, 30, 0)
        prev_ok = c3.number_input("Previous bookings not canceled", 0, 80, 0)

        submitted = st.form_submit_button("Predict", type="primary")

    if not submitted:
        return None

    return {
        "LeadTime": int(lead_time),
        "ArrivalDateYear": arrival.year,
        "ArrivalDateMonth": arrival.strftime("%B"),
        "ArrivalDateWeekNumber": int(arrival.isocalendar()[1]),
        "ArrivalDateDayOfMonth": arrival.day,
        "StaysInWeekendNights": int(weekend_nights),
        "StaysInWeekNights": int(week_nights),
        "Adults": int(adults),
        "Children": int(children),
        "Babies": int(babies),
        "Meal": meal,
        "ReservedRoomType": room,
        "DepositType": deposit,
        "Country": country,
        "MarketSegment": market,
        "DistributionChannel": channel,
        "Agent": agent,
        "CustomerType": customer,
        "IsRepeatedGuest": int(repeated),
        "PreviousCancellations": int(prev_cancel),
        "PreviousBookingsNotCanceled": int(prev_ok),
    }


def choose_model(kind):
    """Let the user pick one of the saved models of this kind."""
    names = [name for name in saved_artifacts() if name.endswith(kind)]
    if not names:
        st.warning("No saved models. Go to the **Models** page and train them first.")
        return None
    default = names.index(f"final_{kind}") if f"final_{kind}" in names else 0
    return st.selectbox(
        "Model",
        names,
        index=default,
        format_func=lambda n: nice_model_name(n.split("_")[0]),
    )


def show_model_info(task, model_name):
    """Validation metrics of the selected model, when its table exists."""
    prefix = model_name.split("_")[0]
    if prefix == "final":
        st.caption("Final model: the best model on the validation set, chosen in `main.py`.")
        return
    table = read_table(f"{prefix}_{task}_baseline.csv")
    if table is not None:
        st.caption("Validation results of this model")
        st.dataframe(table, hide_index=True)


def page_predict_adr():
    st.title(" Predict ADR")
    st.write("Enter a booking to predict its average daily rate.")
    model_name = choose_model("regressor")
    if model_name is None:
        return
    show_model_info("regression", model_name)

    booking = booking_form("adr_form")
    if booking:
        model = cached_model(model_name)
        adr = float(model.predict(prepare_booking(booking))[0])
        st.metric("Predicted ADR", f"€{adr:,.2f} per night")
        nights = booking["StaysInWeekendNights"] + booking["StaysInWeekNights"]
        if nights:
            st.caption(f"About €{adr * nights:,.2f} for {nights} nights.")


def page_predict_cancellation():
    st.title(" Predict cancellation")
    st.write("Enter a booking to predict whether it will be canceled.")
    model_name = choose_model("classifier")
    if model_name is None:
        return
    show_model_info("classification", model_name)

    booking = booking_form("cancel_form")
    if booking:
        model = cached_model(model_name)
        probability = float(model.predict_proba(prepare_booking(booking))[0, 1])
        label = "Likely to cancel" if probability >= 0.5 else "Likely to stay"
        c1, c2 = st.columns(2)
        c1.metric("Cancellation probability", f"{probability:.1%}")
        c2.metric("Prediction (threshold 50%)", label)
        st.progress(probability)


def page_clustering():
    st.title(" Clustering")
    st.write(
        "Bookings are grouped by `LeadTime`, `ADR`, `Adults`, `Children` and `TotalNights`. "
        "No labels are used to build the clusters."
    )

    tab_gmm, tab_kmeans = st.tabs(["GMM", "K-Means"])

    with tab_gmm:
        comparison = read_table("gmm_baseline_vs_optimized.csv")
        if comparison is None:
            st.info("Run `python -m src.models.clustering.gmm_model` first.")
        else:
            st.subheader("Baseline vs optimized")
            st.dataframe(
                comparison[["configuration", "n_components", "covariance_type",
                            "silhouette", "davies_bouldin"]].round(3),
                
                hide_index=True,
            )
            col1, col2 = st.columns(2)
            with col1:
                show_figure("gmm_clusters_pca.png", "Final clusters (PCA 2D)")
            with col2:
                show_figure("gmm_silhouette_dbi.png", "Silhouette and Davies-Bouldin")
                show_figure("gmm_bic.png", "BIC")

            profiles = read_table("gmm_cluster_profiles.csv")
            if profiles is not None:
                st.subheader("Cluster profiles")
                st.caption("Cancel rate is calculated after clustering, only to describe the clusters.")
                st.dataframe(profiles, hide_index=True)

    with tab_kmeans:
        results = read_table("kmeans_evaluation_experiment3.csv")
        if results is None:
            st.info("No K-Means results yet.")
        else:
            st.dataframe(results.round(3), hide_index=True)
        col1, col2 = st.columns(2)
        with col1:
            show_figure("kmeans_clusters_experiment3_k6.png", "K-Means clusters (k=6)")
        with col2:
            show_figure("kmeans_elbow_experiment3.png", "Elbow method")


# ======================================================================
# Navigation
# ======================================================================

PAGES = {
    "Overview": page_overview,
    "Models": page_models,
    "Model comparison": page_comparison,
    "Tuning results": page_tuning,
    "Predict ADR": page_predict_adr,
    "Predict cancellation": page_predict_cancellation,
    "Clustering": page_clustering,
}

st.sidebar.title("City Hotel ML")
choice = st.sidebar.radio("Page", list(PAGES.keys()))
st.sidebar.caption("CSE382 — Fall 2026")
PAGES[choice]()