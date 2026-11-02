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
from src.features.feature_selection import (
    SET_A_FEATURES,
    SET_B_FEATURES,
    SET_C_FEATURES,
    SET_D_FEATURES,
)
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


def train_xgb_classification(config, feature_set="A"):
    feature_sets = {
        "A": SET_A_FEATURES,
        "B": SET_B_FEATURES,
        "C": SET_C_FEATURES,
        "D": SET_D_FEATURES,
    }

    if feature_set not in feature_sets:
        raise ValueError("feature_set must be A, B, C, or D")

    df = clean_data(load_raw_data(), task="classification")

    X = df[feature_sets[feature_set]].copy()
    y = df["IsCanceled"].copy()

    # Shared stratified split: final test set stays untouched
    X_dev, X_test, y_dev, y_test = split_data(
        X, y, stratify=True
    )

    # Same stratified validation split for every feature set
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

    print(f"\nTraining XGBoost with Set {feature_set}...")
    model.fit(X_train, y_train)

    # Predict validation labels and cancellation probabilities
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    results = pd.DataFrame([{
        "model": "XGBoost Classifier",
        "configuration": "untuned",
        "feature_set": feature_set,
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

    suffix = (
        "" if feature_set == "A"
        else f"_set_{feature_set.lower()}"
    )

    results_path = (
        output_dir / f"xgb_classification_baseline{suffix}.csv"
    )

    results.to_csv(results_path, index=False)

    confusion_table = pd.DataFrame(
        cm,
        index=["actual_not_canceled", "actual_canceled"],
        columns=["predicted_not_canceled", "predicted_canceled"],
    )

    confusion_table.to_csv(
        output_dir / f"xgb_classification_confusion_matrix{suffix}.csv",
        index_label="actual_class",
    )

    print(results.round(4).to_string(index=False))

    print("\nConfusion Matrix:")
    print("Rows = actual, columns = predicted; class order = [0, 1]")
    print(cm)

    print(f"\nSaved: {results_path}")

    return model, results


def compare_feature_sets(config):
    all_results = []

    for feature_set in ("A", "B", "C", "D"):
        _, results = train_xgb_classification(
            config,
            feature_set=feature_set,
        )
        all_results.append(results)

    comparison = pd.concat(all_results, ignore_index=True)

    output_dir = PROJECT_ROOT / config["tables_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(
        output_dir / "xgb_classification_feature_comparison.csv",
        index=False,
    )

    columns = [
        "feature_set",
        "features",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
    ]

    print("\nSet A vs B vs C vs D — Validation Results:")
    print(comparison[columns].round(4).to_string(index=False))

    return comparison


if __name__ == "__main__":
    train_xgb_classification(load_config())