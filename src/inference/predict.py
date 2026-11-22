import json

import joblib
import pandas as pd

from src.features.feature_selection import SET_A_FEATURES
from src.utils.config import PROJECT_ROOT, load_config


# ===== Saving and loading models =====

def _artifacts_dir(config=None):
    config = config or load_config()
    path = PROJECT_ROOT / config["artifacts_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_model(model, name, config=None):
    """Save a fitted pipeline to artifacts/<name>.joblib."""
    path = _artifacts_dir(config) / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"Model saved: {path}")
    return path


def load_model(name, config=None):
    """Load a saved pipeline from artifacts/<name>.joblib."""
    path = _artifacts_dir(config) / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\nRun: python main.py train --task all"
        )
    return joblib.load(path)


# ===== Preparing one booking =====

def _clean_agent(value):
    """Match clean_data: missing agent -> 'NoAgent', IDs stored like '9'."""
    if value is None:
        return "NoAgent"
    text = str(value).strip()
    if text.upper() in ("", "NULL", "NAN", "NOAGENT"):
        return "NoAgent"
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def prepare_booking(booking):
    """Turn one booking (dict) into a one-row DataFrame with the Set A columns."""
    missing = [col for col in SET_A_FEATURES if col not in booking]
    if missing:
        raise ValueError(f"Missing booking fields: {missing}")

    df = pd.DataFrame([{col: booking[col] for col in SET_A_FEATURES}])

    # Same rules as clean_data
    df["Agent"] = df["Agent"].map(_clean_agent)
    df["Children"] = pd.to_numeric(df["Children"], errors="coerce").fillna(0)
    df["Country"] = df["Country"].fillna("Unknown")

    return df


# ===== Predictions =====

def predict_adr(booking, model_name="final_regressor"):
    """Predicted average daily rate for one booking."""
    model = load_model(model_name)
    return float(model.predict(prepare_booking(booking))[0])


def predict_cancellation(booking, model_name="final_classifier"):
    """Probability (0-1) that one booking will be canceled."""
    model = load_model(model_name)
    return float(model.predict_proba(prepare_booking(booking))[0, 1])


# ===== Example booking =====

def load_booking_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def make_example_booking(output_path="examples/booking.json"):
    """Save the first real booking from the cleaned data as a JSON example."""
    from src.data.load import load_raw_data
    from src.preprocessing.pipeline import clean_data

    df = clean_data(load_raw_data(), task="regression")
    booking = json.loads(df.iloc[[0]][SET_A_FEATURES].to_json(orient="records"))[0]

    path = PROJECT_ROOT / output_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(booking, file, indent=2)

    print(f"Example booking saved: {path}")
    return booking


if __name__ == "__main__":
    print(json.dumps(make_example_booking(), indent=2))