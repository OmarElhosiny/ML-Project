from src.data.load import load_raw_data
from src.preprocessing.pipeline import clean_data
from src.features.feature_selection import SET_A_FEATURES

df = load_raw_data()
df = clean_data(df, task="regression")

experiment_features = SET_A_FEATURES + [
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth"
]

X = df[experiment_features].copy()
y = df["ADR"].copy()

print("Input shape:", X.shape)
print("Target:", y.name)
print("Missing target values:", y.isna().sum())
from sklearn.model_selection import train_test_split

# Reserve 20% for the final test
X_development, X_test, y_development, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# Split the remaining data into training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_development, y_development,
    test_size=0.25,
    random_state=42
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))
from sklearn.pipeline import Pipeline
from src.preprocessing.transform import create_preprocessor
from src.models.regression.xgb_model import create_xgb_regressor

model = Pipeline([
    ("preprocessor", create_preprocessor(X_train)),
    ("regressor", create_xgb_regressor())
])

print("Training XGBoost...")

model.fit(X_train, y_train)

print("Training completed.")
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Predict ADR for validation bookings
y_pred = model.predict(X_val)

# Compare predictions with actual ADR
mae = mean_absolute_error(y_val, y_pred)
rmse = mean_squared_error(y_val, y_pred) ** 0.5
r2 = r2_score(y_val, y_pred)

print("\nValidation Results:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")
from pathlib import Path
import pandas as pd

output_dir = Path("outputs/tables")
output_dir.mkdir(parents=True, exist_ok=True)

results = pd.DataFrame([{
    "model": "XGBoost Regressor",
    "configuration": "untuned",
    "features": X.shape[1],
    "evaluation_set": "validation",
    "random_state": 42,
    "training_rows": len(X_train),
    "validation_rows": len(X_val),
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}])

results.to_csv(
    output_dir / "xgb_regression_baseline.csv",
    index=False
)

print("Results saved to outputs/tables/xgb_regression_baseline.csv")
examples = pd.DataFrame({
    "Actual ADR": y_val.to_numpy(),
    "Predicted ADR": y_pred
})

examples["Absolute Error"] = (
    examples["Actual ADR"] - examples["Predicted ADR"]
).abs()

print(examples.head(10).round(2).to_string(index=False))