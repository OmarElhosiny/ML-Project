import argparse

from src.utils.config import PROJECT_ROOT, load_config


def check_data(config, args):
    """Check whether the configured raw dataset exists."""
    data_path = PROJECT_ROOT / config["raw_data_path"]
    if data_path.exists():
        print(f"Dataset found: {data_path}")
    else:
        print(f"Dataset not found: {data_path}")
        print("Place the dataset at the configured path.")


def run_analysis(config, args):
    """Placeholder for EDA (member 4)."""
    print("EDA is not implemented yet.")


def run_preprocessing(config, args):
    """Placeholder for the preprocessing pipeline (member 1)."""
    print("Preprocessing pipeline is not implemented yet.")


def run_training(config, args):
    """Train supervised models and save them to artifacts/."""
    from src.inference.predict import save_model

    if args.task in ("regression", "all"):
        from src.models.regression.xgb_model import train_xgb_regression
        from src.models.regression.random_forest_model import train_rf_regression

        xgb_model, _ = train_xgb_regression(config)
        rf_model, _ = train_rf_regression(config)
        save_model(xgb_model, "xgb_regressor", config)
        save_model(rf_model, "rf_regressor", config)

        # Best baseline on validation (update after tuning)
        save_model(rf_model, "final_regressor", config)

    if args.task in ("classification", "all"):
        from src.models.classification.xgb_model import train_xgb_classification
        from src.models.classification.random_forest_model import train_rf_classification

        xgb_model, _ = train_xgb_classification(config)
        rf_model, _ = train_rf_classification(config)
        save_model(xgb_model, "xgb_classifier", config)
        save_model(rf_model, "rf_classifier", config)

        # Best baseline F1 on validation (update after tuning)
        save_model(rf_model, "final_classifier", config)


def run_clustering(config, args):
    """Placeholder for clustering (members 4, 5, 6)."""
    print("Clustering is not implemented yet.")


def run_evaluation(config, args):
    """Placeholder for model evaluation (member 2)."""
    print("Model evaluation is not implemented yet.")


def run_prediction(config, args):
    """Predict for one booking from a JSON file using the saved final models."""
    from src.inference.predict import (
        load_booking_json,
        predict_adr,
        predict_cancellation,
    )

    booking = load_booking_json(args.input)

    if args.task == "regression":
        print(f"Predicted ADR: {predict_adr(booking):.2f}")
    else:
        probability = predict_cancellation(booking)
        label = "Canceled" if probability >= 0.5 else "Not canceled"
        print(f"Cancellation probability: {probability:.2%} ({label})")


def main():
    parser = argparse.ArgumentParser(description="City Hotel (H2) ML Project CLI")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to the YAML configuration file")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check-data", help="Check whether the raw dataset exists")
    sub.add_parser("analyze", help="Run EDA and save figures/tables")
    sub.add_parser("preprocess", help="Run the preprocessing pipeline")

    train = sub.add_parser("train", help="Train and save supervised models")
    train.add_argument("--task", choices=["regression", "classification", "all"],
                       default="all")

    sub.add_parser("cluster", help="Run clustering algorithms")
    sub.add_parser("evaluate", help="Build comparison tables")

    predict = sub.add_parser("predict", help="Predict for one booking")
    predict.add_argument("--task", choices=["regression", "classification"],
                         required=True)
    predict.add_argument("--input", required=True,
                         help="Path to a JSON file with booking features")

    args = parser.parse_args()
    config = load_config(args.config)

    commands = {
        "check-data": check_data,
        "analyze": run_analysis,
        "preprocess": run_preprocessing,
        "train": run_training,
        "cluster": run_clustering,
        "evaluate": run_evaluation,
        "predict": run_prediction,
    }
    commands[args.command](config, args)


if __name__ == "__main__":
    main()