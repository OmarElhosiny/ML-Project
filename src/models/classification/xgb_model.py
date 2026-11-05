import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.load import load_raw_data
from src.features.feature_selection import SET_A_FEATURES
from src.preprocessing.pipeline import clean_data
from src.preprocessing.split import split_data
from src.preprocessing.transform import create_preprocessor
from src.utils.config import PROJECT_ROOT, load_config


def create_xgb_classifier(config):
    return XGBClassifier(
        objective="binary:logistic",
        random_state=config["random_state"],
        n_jobs=-1,
    )


def train_xgb_classification(config):
    df = clean_data(load_raw_data(), task="classification")

    X = df[SET_A_FEATURES].copy()
    y = df["IsCanceled"].copy()

    # Shared stratified split: keep the final test set untouched
    X_dev, X_test, y_dev, y_test = split_data(
        X, y, stratify=True
    )

    # Create stratified training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.25,
        random_state=config["random_state"],
        stratify=y_dev,
    )

    model = Pipeline([
        ("preprocessor", create_preprocessor(X_train)),
        ("classifier", create_xgb_classifier(config)),
    ])

    print("Training XGBoost Classifier with Set A...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    results = pd.DataFrame([{
        "model": "XGBoost Classifier",
        "configuration": "untuned",
        "feature_set": "A",
        "features": X.shape[1],
        "evaluation_set": "validation",
        "split_strategy": "stratified",
        "random_state": config["random_state"],
        "training_rows": len(X_train),
        "validation_rows": len(X_val),
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(
            y_val, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_val, y_pred, zero_division=0
        ),
        "F1": f1_score(y_val, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_val, y_prob),
    }])

    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])

    output_dir = PROJECT_ROOT / config["tables_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "xgb_classification_baseline.csv"
    results.to_csv(results_path, index=False)

    confusion_table = pd.DataFrame(
        cm,
        index=["actual_not_canceled", "actual_canceled"],
        columns=["predicted_not_canceled", "predicted_canceled"],
    )

    confusion_table.to_csv(
        output_dir / "xgb_classification_confusion_matrix.csv",
        index_label="actual_class",
    )

    print(results.round(4).to_string(index=False))

    print("\nConfusion Matrix:")
    print("Rows = actual, columns = predicted; class order = [0, 1]")
    print(cm)

    print(f"\nSaved: {results_path}")

    return model, results


if __name__ == "__main__":
    train_xgb_classification(load_config())