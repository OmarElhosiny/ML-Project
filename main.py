import argparse
from pathlib import Path

import yaml


def load_config(config_path):
    """Load project settings from a YAML file."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML dictionary.")

    return config


def check_data(config):
    """Check whether the configured raw dataset exists."""
    data_path = Path(config["raw_data_path"])

    if data_path.exists():
        print(f"Dataset found: {data_path}")
    else:
        print(f"Dataset not found: {data_path}")
        print("Place the dataset at the configured path.")


def run_preprocessing(config):
    """Placeholder for the preprocessing pipeline."""
    print("Preprocessing pipeline is not implemented yet.")
    print(f"Input: {config['raw_data_path']}")
    print(f"Output directory: {config['processed_dir']}")


def run_training(config):
    """Placeholder for model training."""
    print("Model training is not implemented yet.")
    print(f"Random state: {config['random_state']}")


def run_evaluation(config):
    """Placeholder for model evaluation."""
    print("Model evaluation is not implemented yet.")
    print(f"Figures directory: {config['figures_dir']}")
    print(f"Tables directory: {config['tables_dir']}")


def main():
    parser = argparse.ArgumentParser(
        description="Machine Learning Project CLI"
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "check-data",
        help="Check whether the raw dataset exists",
    )
    subparsers.add_parser(
        "preprocess",
        help="Run the preprocessing pipeline",
    )
    subparsers.add_parser(
        "train",
        help="Train machine learning models",
    )
    subparsers.add_parser(
        "evaluate",
        help="Evaluate trained models",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    commands = {
        "check-data": check_data,
        "preprocess": run_preprocessing,
        "train": run_training,
        "evaluate": run_evaluation,
    }

    commands[args.command](config)


if __name__ == "__main__":
    main()
