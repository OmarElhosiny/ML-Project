import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.data.load import load_raw_data
from src.features.feature_selection import SET_A_FEATURES
from src.preprocessing.pipeline import clean_data
from src.preprocessing.split import split_data
from src.preprocessing.transform import create_preprocessor
from src.utils.config import PROJECT_ROOT, load_config


def create_xgb_regressor(config):
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=config["random_state"],
        n_jobs=-1,
    )


def train_xgb_regression(config):
    df = clean_data(load_raw_data(), task="regression")

    X = df[SET_A_FEATURES].copy()
    y = df["ADR"].copy()

    # Use the team's shared split; keep the test set untouched
    X_dev, X_test, y_dev, y_test = split_data(X, y)

    # Create validation data from the development set
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.25,
        random_state=config["random_state"],
    )

    model = Pipeline([
        ("preprocessor", create_preprocessor(X_train)),
        ("regressor", create_xgb_regressor(config)),
    ])

    print("Training XGBoost...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    results = pd.DataFrame([{
        "model": "XGBoost Regressor",
        "configuration": "untuned",
        "features": X.shape[1],
        "evaluation_set": "validation",
        "random_state": config["random_state"],
        "training_rows": len(X_train),
        "validation_rows": len(X_val),
        "MAE": mean_absolute_error(y_val, y_pred),
        "RMSE": mean_squared_error(y_val, y_pred) ** 0.5,
        "R2": r2_score(y_val, y_pred),
    }])

    output_path = (
        PROJECT_ROOT
        / config["tables_dir"]
        / "xgb_regression_baseline.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(results.round(4).to_string(index=False))
    print(f"Saved: {output_path}")

    return model, results


if __name__ == "__main__":
    train_xgb_regression(load_config())