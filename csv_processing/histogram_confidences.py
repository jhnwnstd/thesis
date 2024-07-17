import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define dataset paths
DATASET_PATHS: Dict[str, str] = {
    "CLMET3": 'main/data/outputs/csv/sorted_tokens_clmet_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Lampeter": 'main/data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Edges": 'main/data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "CMU": 'main/data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Brown": 'main/data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv'
}

def load_dataset(args: Tuple[str, str]) -> Tuple[str, Optional[pd.DataFrame]] :
    """Loads dataset from the given path, handling errors."""
    name, path = args
    try:
        logger.info(f"Loading dataset from {path}")
        df = pd.read_csv(path, usecols=['Top1_Confidence', 'Top1_Is_Valid', 'Top1_Is_Accurate'])
        return name, df
    except Exception as e:
        logger.error(f"Error loading dataset {name} from {path}: {e}")
        return name, None

def calculate_histogram_data(df: pd.DataFrame, column_name: str, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray] :
    """Calculate histogram data for the given dataframe and column."""
    valid_predictions = df[df[column_name]]["Top1_Confidence"]
    invalid_predictions = df[~df[column_name]]["Top1_Confidence"]
    
    valid_counts, _ = np.histogram(valid_predictions, bins=bins)
    invalid_counts, _ = np.histogram(invalid_predictions, bins=bins)
    
    total_counts = valid_counts + invalid_counts
    valid_proportions = np.divide(valid_counts, total_counts, where=total_counts != 0)
    invalid_proportions = np.divide(invalid_counts, total_counts, where=total_counts != 0)
    
    return valid_proportions, invalid_proportions, total_counts

def plot_normalized_stacked_histogram(ax: plt.Axes, dataset: pd.DataFrame, valid_color: str, invalid_color: str, 
                                      label: str, column_name: str, threshold: float = 0.60, bins: int = 30) -> None:
    """Enhanced plotting to highlight the first bin where proportions exceed a given threshold."""
    if dataset is None or dataset.empty:
        ax.text(0.5, 0.5, 'Data Unavailable', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16)
        return

    bins = np.linspace(0, 1, bins + 1)
    valid_proportions, invalid_proportions, total_counts = calculate_histogram_data(dataset, column_name, bins)
    
    # Find the first bin where valid proportions exceed the threshold
    first_threshold_bin_index = np.argmax(valid_proportions >= threshold)
    first_threshold_bin = bins[first_threshold_bin_index]
    
    ax.bar(bins[:-1], valid_proportions, width=np.diff(bins), align='edge', color=valid_color, alpha=0.75, label='Valid' if column_name == "Top1_Is_Valid" else 'Accurate')
    ax.bar(bins[:-1], invalid_proportions, width=np.diff(bins), align='edge', color=invalid_color, alpha=0.65, label='Invalid' if column_name == "Top1_Is_Valid" else 'Inaccurate', bottom=valid_proportions)
    
    # Highlight the bin where valid predictions first exceed the threshold
    if valid_proportions[first_threshold_bin_index] >= threshold:
        threshold_value = valid_proportions[first_threshold_bin_index]
        ax.axvline(first_threshold_bin, color='black', linestyle='--')
        ax.annotate(f'{first_threshold_bin:.2f}', xy=(first_threshold_bin, threshold_value), xytext=(first_threshold_bin + 0.05, threshold_value - 0.05),
                    arrowprops=dict(facecolor='black', shrink=0.05), fontsize=16, color='black', fontweight='bold')
        ax.plot([], [], color='black', linestyle='--', label=f'Threshold at {threshold*100:.0f}%')

    ax.set_xlabel('Top 1 Confidence', fontsize=14)
    ax.set_ylabel('Proportion', fontsize=14)
    ax.set_title(f'{label} Dataset ({column_name.split("_")[-1].capitalize()})', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.minorticks_on()

def plot_and_save_figures(loaded_datasets: Dict[str, pd.DataFrame], column_name: str, filename: str) -> None:
    """Plot and save figures for given column_name."""
    n_datasets = len(loaded_datasets)
    n_cols = 2
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows), squeeze=False)
    colors = [plt.get_cmap('tab10')(i) for i in range(n_datasets)]

    combined_data = []

    for (label, dataset), color, ax in zip(loaded_datasets.items(), colors, axs.flatten()):
        plot_normalized_stacked_histogram(ax, dataset, color, 'tab:gray', label, column_name)
        if dataset is not None and not dataset.empty:
            combined_data.append(dataset)

    for ax in axs.flatten()[len(loaded_datasets):]:
        ax.set_visible(False)

    plt.tight_layout(pad=2.0)
    output_dir = Path('output/confs')
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f'{filename}.png')
    plt.close(fig)

    # Combine all datasets and plot the combined histogram
    if combined_data:
        combined_dataset = pd.concat(combined_data, ignore_index=True)
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_normalized_stacked_histogram(ax, combined_dataset, 'tab:blue', 'tab:gray', 'Combined', column_name)
        fig.savefig(output_dir / f'combined_{filename}.png')
        plt.close(fig)

def main():
    plt.style.use('seaborn-v0_8-colorblind')

    # Load datasets in parallel
    with ProcessPoolExecutor() as executor:
        future_to_dataset = {executor.submit(load_dataset, (name, path)): name for name, path in DATASET_PATHS.items()}
        loaded_datasets = {}
        for future in as_completed(future_to_dataset):
            name, df = future.result()
            if df is not None:
                loaded_datasets[name] = df

    # Create plots for "Top1_Is_Accurate"
    plot_and_save_figures(loaded_datasets, "Top1_Is_Accurate", 'normalized_accurate_stacked_histograms')

    # Create plots for "Top1_Is_Valid"
    plot_and_save_figures(loaded_datasets, "Top1_Is_Valid", 'normalized_valid_stacked_histograms')

if __name__ == "__main__":
    main()
