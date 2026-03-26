import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from config import DATASET_PATHS
from pygam import LogisticGAM, s  # type: ignore[import-not-found]
from sklearn.metrics import (  # type: ignore[import-not-found,import-untyped]
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(filepath: Path) -> Optional[pd.DataFrame]:
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully from {filepath}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return None


def prepare_data(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Prepare data by calculating word length and normalized missing letter index."""
    required_columns = {"Top1_Is_Accurate", "Tested_Word"}

    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        logger.error(f"Required columns are missing: {missing_columns}")
        return None

    data["Word_Length"] = data["Tested_Word"].str.len()
    data["Normalized_Missing_Index"] = data["Tested_Word"].str.find("_") / (
        data["Word_Length"] - 1
    )
    data = data.replace(
        {"Normalized_Missing_Index": {np.inf: np.nan, -np.inf: np.nan}}
    ).dropna()

    return data


def fit_model(
    X: "pd.DataFrame | np.ndarray",
    y: "pd.Series | np.ndarray",
    n_splines: int = 15,
) -> Optional[LogisticGAM]:
    """Fit a logistic GAM model."""
    try:
        gam = LogisticGAM(s(0, n_splines=n_splines)).fit(X, y)
        logger.info("Model fitting complete")
        return gam
    except Exception as e:
        logger.error(f"Error fitting model: {str(e)}")
        return None


def plot_results(
    XX: np.ndarray,
    proba: np.ndarray,
    X: np.ndarray,
    y: "np.ndarray | pd.Series",
    title: str,
    config: dict,
    output_path: Path,
):
    """Plot the results of the GAM model predictions against the actual data."""
    plt.figure(figsize=config["figsize"])
    sns.set_style("whitegrid")
    plt.plot(
        XX,
        proba,
        label="Model Prediction",
        color=config["prediction_color"],
        linewidth=2,
    )
    plt.scatter(
        X, y, color=config["data_color"], alpha=0.7, label="Actual Data"
    )
    plt.xlabel("Normalized Missing Index", fontsize=12)
    plt.ylabel("Prediction Accuracy", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.xticks(
        np.arange(0, 1.1, 0.1),
        labels=[f"{tick:.1f}" for tick in np.arange(0, 1.1, 0.1)],
    )

    if config["dynamic_range"]:
        center_point = np.median(proba)
        margin = 0.30
        plt.ylim(
            [max(0, center_point - margin), min(1, center_point + margin)]
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def calculate_metrics(
    y_true: "np.ndarray | pd.Series", y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate various performance metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def process_dataset(
    args: Tuple[str, Path, dict],
) -> Optional[
    Tuple[
        str,
        np.ndarray,
        "np.ndarray | pd.Series",
        np.ndarray,
        np.ndarray,
        Dict[str, float],
    ]
]:
    """Process each dataset: load data, prepare it, fit the model, plot results, and calculate metrics."""
    name, path, config = args
    data = load_data(path)
    if data is None:
        return None

    prepared_data = prepare_data(data)
    if prepared_data is None:
        return None

    X = prepared_data[["Normalized_Missing_Index"]]
    y = prepared_data["Top1_Is_Accurate"]
    gam = fit_model(X, y)
    if gam is None:
        return None

    XX = np.linspace(0, 1, 1000)[:, None]
    proba = gam.predict_proba(XX)
    output_path = Path("output/gams") / f"{name}_GAM_df.png"
    plot_results(
        XX.ravel(),
        proba,
        X.to_numpy().ravel(),
        y,
        f"Effect of Normalized Missing Index on Prediction Accuracy in {name}",
        config,
        output_path,
    )

    y_pred = gam.predict(X) > 0.5
    metrics = calculate_metrics(y, y_pred)

    return name, X.to_numpy().ravel(), y, XX.ravel(), proba, metrics


def plot_all_datasets(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    config: dict,
    output_path: Path,
):
    """Plot the results of all datasets on a single graph."""
    plt.figure(figsize=config["figsize"])
    sns.set_style("whitegrid")

    for name, (X, y, XX, proba) in datasets.items():
        plt.plot(XX, proba, label=f"{name} Model Prediction", linewidth=2)
        plt.scatter(X, y, alpha=0.7, label=f"{name} Actual Data")

    plt.xlabel("Normalized Missing Index", fontsize=12)
    plt.ylabel("Prediction Accuracy", fontsize=12)
    plt.title(
        "Effect of Normalized Missing Index on Prediction Accuracy across Datasets",
        fontsize=14,
    )
    plt.legend()
    plt.xticks(
        np.arange(0, 1.1, 0.1),
        labels=[f"{tick:.1f}" for tick in np.arange(0, 1.1, 0.1)],
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def process_combined_dataset(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    config: dict,
) -> Optional[Dict[str, float]]:
    """Process all datasets, combine them, fit a single model, plot the results, and return metrics."""
    if not datasets:
        logger.error("No datasets to combine")
        return None

    combined_X = np.concatenate([X for X, _, _, _ in datasets.values()])
    combined_y = np.concatenate([y for _, y, _, _ in datasets.values()])

    gam = fit_model(combined_X.reshape(-1, 1), combined_y)
    if gam is None:
        return None

    XX = np.linspace(0, 1, 1000)[:, None]
    proba = gam.predict_proba(XX)
    output_path = Path("output/gams/combined_dataset_GAM_df.png")
    plot_results(
        XX.ravel(),
        proba,
        combined_X,
        combined_y,
        "Effect of Normalized Missing Index on Prediction Accuracy for Combined Dataset",
        config,
        output_path,
    )

    y_pred = gam.predict(combined_X.reshape(-1, 1)) > 0.5
    return calculate_metrics(combined_y, y_pred)


def main():
    default_plot_config = {
        "figsize": (14, 8),
        "prediction_color": "blue",
        "data_color": "black",
        "dynamic_range": True,
    }

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_dataset, (name, path, default_plot_config))
            for name, path in DATASET_PATHS.items()
        ]
        datasets = {}
        all_metrics = {}
        for future in as_completed(futures):
            result = future.result()
            if result:
                name, X, y, XX, proba, metrics = result
                datasets[name] = (X, y, XX, proba)
                all_metrics[name] = metrics

    plot_all_datasets(
        datasets,
        default_plot_config,
        Path("output/gams/all_datasets_GAM_df.png"),
    )

    combined_metrics = process_combined_dataset(datasets, default_plot_config)

    for name, metrics in all_metrics.items():
        print(f"\nMetrics for {name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    if combined_metrics:
        print("\nMetrics for Combined Dataset:")
        for metric, value in combined_metrics.items():
            print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
