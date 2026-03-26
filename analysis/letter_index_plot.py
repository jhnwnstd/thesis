import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from config import DATASET_PATHS
from utils import (
    calculate_missing_index_stats,
    load_and_prepare_index_data,
    perform_linear_regression,
    perform_logistic_regression,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

INDEX_FORMULA = "Top1_Is_Accurate ~ Normalized_Missing_Index"


def plot_accuracy_vs_index(stats: pd.DataFrame, dataset_name: str):
    """Create a plot of accuracy vs normalized missing index."""
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=stats,
        x="Normalized_Missing_Index",
        y="Accuracy_Mean",
        size="Sample_Count",
        legend=False,
    )

    # Filter out rows with NaN in Accuracy_SEM
    valid_stats = stats.dropna(subset=["Accuracy_SEM"])

    plt.errorbar(
        valid_stats["Normalized_Missing_Index"],
        valid_stats["Accuracy_Mean"],
        yerr=valid_stats["Accuracy_SEM"],
        fmt="none",
        alpha=0.5,
    )

    # Add trend line using linear regression
    sns.regplot(
        x="Normalized_Missing_Index",
        y="Accuracy_Mean",
        data=stats,
        scatter=False,
        color="red",
    )

    plt.title(
        f"Accuracy vs Normalized Missing Index for {dataset_name} Dataset"
    )
    plt.xlabel("Normalized Missing Index")
    plt.ylabel("Accuracy")

    # Define the save path for the plot
    save_dir = Path("output/letter_index_plots")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{dataset_name}_accuracy_vs_index.png"

    plt.savefig(save_path)
    plt.close()


def analyze_dataset(dataset_name: str, file_path: Path):
    """Analyze a single dataset."""
    logger.info(f"\nProcessing dataset: {dataset_name}")
    logger.info("=" * 50)
    data = load_and_prepare_index_data(file_path)
    if data is None:
        return

    stats = calculate_missing_index_stats(data)
    logit_model = perform_logistic_regression(data, INDEX_FORMULA)
    ols_model = perform_linear_regression(data, INDEX_FORMULA)
    plot_accuracy_vs_index(stats, dataset_name)

    logger.info(f"\n{dataset_name} Dataset Analysis:")
    logger.info("=" * 50)
    logger.info(
        "Statistical Analysis of Normalized Missing Index and Prediction Accuracy:"
    )
    logger.info(f"\n{stats}\n")
    logger.info("Data Types:")
    logger.info(f"{data.dtypes}\n")
    if logit_model is not None:
        logger.info("Logistic Regression Analysis Summary:")
        logger.info(f"\n{logit_model.summary()}\n")
    else:
        logger.info("Logistic Regression Analysis failed.\n")
    if ols_model is not None:
        logger.info("Linear Regression Analysis Summary:")
        logger.info(f"\n{ols_model.summary()}\n")
    else:
        logger.info("Linear Regression Analysis failed.\n")


def main():
    for name, path in DATASET_PATHS.items():
        analyze_dataset(name, path)


if __name__ == "__main__":
    main()
