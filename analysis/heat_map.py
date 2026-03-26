from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from config import DATASET_PATHS

# Set a color-blind friendly style for plots
plt.style.use("seaborn-v0_8-colorblind")


def load_dataset(filepath: Path) -> pd.DataFrame:
    """Load dataset from a specified filepath."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return pd.DataFrame()


def filter_mispredictions(data: pd.DataFrame) -> pd.DataFrame:
    """Filter dataset to only include mispredictions."""
    return (
        data[data["Correct_Letters"] != data["Top1_Predicted_Letter"]]
        if not data.empty
        else pd.DataFrame()
    )


def calculate_confusion_matrix(mispredictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate and normalize the confusion matrix for mispredictions."""
    confusion_matrix = pd.crosstab(
        mispredictions["Correct_Letters"],
        mispredictions["Top1_Predicted_Letter"],
    )
    confusion_matrix = confusion_matrix.astype(float)
    np.fill_diagonal(confusion_matrix.values, np.nan)
    return confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0)


def plot_heatmap(
    confusion_matrix: pd.DataFrame,
    dataset_name: str,
    output_dir: Path,
    threshold: float = 0.15,
    figsize: Tuple[int, int] = (12, 10),
    annot_fmt: str = ".2f",
) -> None:
    """Enhanced heatmap plotting function with customizable parameters and saving to file."""
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=annot_fmt,
        cmap="viridis",
        cbar=True,
        cbar_kws={"label": "Frequency Proportion"},
    )
    plt.title(
        f"Heatmap of Most Common Substitutions for Missed Letters in {dataset_name}",
        fontsize=16,
    )
    plt.xlabel("Predicted Letter", fontsize=14)
    plt.ylabel("Actual Letter", fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    annotate_heatmap(ax, threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{dataset_name}_heatmap.png", dpi=300)
    plt.close()


def annotate_heatmap(ax: sns.heatmap, threshold: float) -> None:
    """Apply conditional formatting to heatmap annotations based on a threshold."""
    for text in ax.texts:
        try:
            t = float(text.get_text())
            if t == 0:
                text.set_text("")
            elif t < threshold:
                text.set_color("gray")
        except ValueError:
            continue


def main():
    output_directory = Path("output/heatmaps")

    # Process each dataset
    all_mispredictions = []
    for name, path in DATASET_PATHS.items():
        data = load_dataset(path)
        mispredictions = filter_mispredictions(data)
        if not mispredictions.empty:
            all_mispredictions.append(mispredictions)
            confusion_matrix = calculate_confusion_matrix(mispredictions)
            plot_heatmap(confusion_matrix, name, output_directory)

    # Combine all mispredictions into one DataFrame
    if all_mispredictions:
        combined_mispredictions = pd.concat(
            all_mispredictions, ignore_index=True
        )
        combined_confusion_matrix = calculate_confusion_matrix(
            combined_mispredictions
        )
        plot_heatmap(
            combined_confusion_matrix, "All_Datasets", output_directory
        )


if __name__ == "__main__":
    main()
