from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from config import DATASET_PATHS


def ensure_directory_exists(directory: Path) -> None:
    """Ensure that the given directory exists, create it if it doesn't."""
    directory.mkdir(parents=True, exist_ok=True)


def plot_predictions(
    data: pd.DataFrame, dataset_name: str, save_path: Path
) -> None:
    """Plot the distribution of predicted letters and their accuracies and save the plot."""
    plt.figure(figsize=(12, 6))

    # Plot total predictions
    predicted_letter_counts = (
        data["Top1_Predicted_Letter"].value_counts().sort_index()
    )
    plt.bar(
        predicted_letter_counts.index,
        predicted_letter_counts.values.tolist(),
        label="Total Predictions",
        alpha=0.7,
    )

    # Plot accurate predictions
    accurate_predicted_letter_counts = (
        data[data["Top1_Is_Accurate"]]["Top1_Predicted_Letter"]
        .value_counts()
        .sort_index()
    )
    plt.bar(
        accurate_predicted_letter_counts.index,
        accurate_predicted_letter_counts.values.tolist(),
        label="Accurate Predictions",
        alpha=0.7,
    )

    plt.title(
        f"Distribution of Predicted Letters and Their Accuracies ({dataset_name})"
    )
    plt.xlabel("Predicted Letter")
    plt.ylabel("Number of Predictions")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()

    ensure_directory_exists(save_path.parent)
    plt.savefig(save_path)
    plt.close()


def load_datasets(paths: dict) -> dict:
    """Load all datasets into a dictionary."""
    return {name: pd.read_csv(path) for name, path in paths.items()}


def main():
    output_dir = Path("output/prediction_plots")
    datasets = load_datasets(DATASET_PATHS)

    # Plot for each individual dataset
    for name, data in datasets.items():
        plot_predictions(data, name, output_dir / f"{name}_predictions.png")

    # Combine all datasets into one DataFrame and plot
    combined_data = pd.concat(datasets.values(), ignore_index=True)
    plot_predictions(
        combined_data,
        "Combined Datasets",
        output_dir / "Combined_Datasets_predictions.png",
    )


if __name__ == "__main__":
    main()
