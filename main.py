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
    """Placeholder for supervised model training."""
    print(f"Training for task '{args.task}' is not implemented yet.")


def run_clustering(config, args):
    """Placeholder for clustering (members 4, 5, 6)."""
    print("Clustering is not implemented yet.")


def run_evaluation(config, args):
    """Placeholder for model evaluation (member 2)."""
    print("Model evaluation is not implemented yet.")


def run_prediction(config, args):
    """Placeholder for inference (member 3)."""
    print(f"Prediction for task '{args.task}' from {args.input} is not implemented yet.")


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