import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional, List

# Configure logging with a specific format to include the timestamp, log level, and message
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define dataset paths in a dictionary, mapping dataset names to their file paths
DATASET_PATHS: Dict[str, Path] = {
    "CLMET3": Path('main/data/outputs/csv/sorted_tokens_clmet_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Lampeter": Path('main/data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Edges": Path('main/data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "CMU": Path('main/data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Brown": Path('main/data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv')
}

def load_dataset(args: Tuple[str, Path]) -> Tuple[str, Optional[pd.DataFrame]]: 
    """
    Loads dataset from the given path, handling errors.

    Args:
        args (Tuple[str, Path]): A tuple containing the dataset name and file path.

    Returns:
        Tuple[str, Optional[pd.DataFrame]]: A tuple containing the dataset name and the loaded DataFrame (or None if loading failed).
    """
    name, path = args
    try:
        logger.info(f"Loading dataset from {path}")
        # Read specific columns from the CSV file
        df = pd.read_csv(path, usecols=['Top1_Confidence', 'Top1_Is_Valid', 'Top1_Is_Accurate'])
        return name, df
    except Exception as e:
        # Log an error if the dataset cannot be loaded
        logger.error(f"Error loading dataset {name} from {path}: {e}")
        return name, None

def calculate_histogram_data(df: pd.DataFrame, column_name: str, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate histogram data for the given dataframe and column.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        column_name (str): The column name to filter the data by.
        bins (np.ndarray): The bin edges for the histogram.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of valid and invalid proportions for each bin.
    """
    # Separate predictions into valid and invalid based on the given column
    valid_predictions = df[df[column_name]]["Top1_Confidence"]
    invalid_predictions = df[~df[column_name]]["Top1_Confidence"]
    
    # Calculate histogram counts for valid and invalid predictions
    valid_counts, _ = np.histogram(valid_predictions, bins=bins)
    invalid_counts, _ = np.histogram(invalid_predictions, bins=bins)
    
    # Calculate total counts and proportions for valid and invalid predictions
    total_counts = valid_counts + invalid_counts
    valid_proportions = np.divide(valid_counts, total_counts, where=total_counts != 0)
    invalid_proportions = np.divide(invalid_counts, total_counts, where=total_counts != 0)
    
    return valid_proportions, invalid_proportions

def plot_histogram(ax: plt.Axes, dataset: pd.DataFrame, valid_color: str, invalid_color: str, 
                   label: str, column_name: str, threshold: float = 0.60, bins: int = 30) -> None:
    """
    Plot normalized stacked histogram and highlight the first threshold bin.

    Args:
        ax (plt.Axes): The Axes object to plot on.
        dataset (pd.DataFrame): The dataset to plot.
        valid_color (str): The color for valid predictions.
        invalid_color (str): The color for invalid predictions.
        label (str): The label for the plot title.
        column_name (str): The column name to filter the data by.
        threshold (float, optional): The threshold for highlighting bins. Defaults to 0.60.
        bins (int, optional): The number of bins for the histogram. Defaults to 30.
    """
    if dataset is None or dataset.empty:
        # If the dataset is empty, display a message on the plot
        ax.text(0.5, 0.5, 'Data Unavailable', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16)
        return

    # Define bin edges for the histogram
    bins = np.linspace(0.1, 1, bins + 1)
    # Calculate valid and invalid proportions
    valid_proportions, invalid_proportions = calculate_histogram_data(dataset, column_name, bins)
    
    # Find the first bin where valid proportions exceed the threshold
    threshold_bin_indices = np.where(valid_proportions >= threshold)[0]
    if len(threshold_bin_indices) > 0:
        first_threshold_bin_index = threshold_bin_indices[0]
        first_threshold_bin = bins[first_threshold_bin_index]
        
        # Highlight the first threshold bin on the plot
        ax.axvline(first_threshold_bin, color='black', linestyle='--')
        ax.annotate(f'{first_threshold_bin:.2f}', xy=(first_threshold_bin, valid_proportions[first_threshold_bin_index]), 
                    xytext=(first_threshold_bin + 0.05, valid_proportions[first_threshold_bin_index] - 0.05),
                    arrowprops=dict(facecolor='black', shrink=0.05), fontsize=16, color='black', fontweight='bold')
        ax.plot([], [], color='black', linestyle='--', label=f'Threshold at {threshold*100:.0f}%')

    # Determine labels for valid and invalid predictions based on the column name
    label_valid = 'Valid' if column_name == "Top1_Is_Valid" else 'Accurate'
    label_invalid = 'Invalid' if column_name == "Top1_Is_Valid" else 'Inaccurate'

    # Plot stacked bar chart for valid and invalid proportions
    ax.bar(bins[:-1], valid_proportions, width=np.diff(bins), align='edge', color=valid_color, alpha=0.75, label=label_valid)
    ax.bar(bins[:-1], invalid_proportions, width=np.diff(bins), align='edge', color=invalid_color, alpha=0.65, label=label_invalid, bottom=valid_proportions)
    
    # Set plot labels and title
    ax.set_xlabel('Top 1 Confidence', fontsize=14)
    ax.set_ylabel('Proportion', fontsize=14)
    ax.set_title(f'{label} Dataset ({label_valid})', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.minorticks_on()

def create_figures_for_datasets(loaded_datasets: Dict[str, Optional[pd.DataFrame]], column_name: str) -> List[plt.Figure]:
    """
    Create figures for the given column name.

    Args:
        loaded_datasets (Dict[str, Optional[pd.DataFrame]]): A dictionary of loaded datasets.
        column_name (str): The column name to filter the data by.

    Returns:
        List[plt.Figure]: A list of created figures.
    """
    n_datasets = len(loaded_datasets)
    n_cols = 2  # Number of columns for subplots
    n_rows = (n_datasets + n_cols - 1) // n_cols  # Calculate the number of rows needed

    # Create a grid of subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows), squeeze=False)
    # Define a color palette for the datasets
    colors = [plt.get_cmap('tab10')(i) for i in range(n_datasets)]
    combined_data = []

    # Plot histograms for each dataset
    for (label, dataset), color, ax in zip(loaded_datasets.items(), colors, axs.flatten()):
        plot_histogram(ax, dataset, color, 'tab:gray', label, column_name)
        if dataset is not None and not dataset.empty:
            combined_data.append(dataset)

    # Hide any unused subplots
    for ax in axs.flatten()[len(loaded_datasets):]:
        ax.set_visible(False)

    plt.tight_layout(pad=2.0)

    combined_fig, combined_ax = None, None
    if combined_data:
        # Combine all datasets and plot the combined histogram
        combined_dataset = pd.concat(combined_data, ignore_index=True)
        combined_fig, combined_ax = plt.subplots(figsize=(12, 8))
        plot_histogram(combined_ax, combined_dataset, 'tab:blue', 'tab:gray', 'Combined', column_name)
    
    return [fig] + ([combined_fig] if combined_fig else [])

def save_figures(figures: List[plt.Figure], filename: str) -> None:
    """
    Save figures to the output directory.

    Args:
        figures (List[plt.Figure]): The list of figures to save.
        filename (str): The base filename to use for saving.
    """
    output_dir = Path('output/confs')
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists
    for i, fig in enumerate(figures):
        # Determine the suffix for combined figures
        suffix = f'_combined' if i > 0 else ''
        fig.savefig(output_dir / f'{filename}{suffix}.png')  # Save the figure
        plt.close(fig)  # Close the figure to free up memory

def main():
    """
    Main function to load datasets, create figures, and save them.
    """
    plt.style.use('seaborn-v0_8-colorblind')  # Set the plot style

    loaded_datasets = {}
    # Load each dataset and store it in the dictionary
    for name, path in DATASET_PATHS.items():
        _, df = load_dataset((name, path))
        if df is not None:
            loaded_datasets[name] = df

    # Create and save figures for each specified column name
    for column_name, filename in [("Top1_Is_Accurate", 'normalized_accurate_stacked_histograms'), 
                                  ("Top1_Is_Valid", 'normalized_valid_stacked_histograms')]:
        figures = create_figures_for_datasets(loaded_datasets, column_name)
        save_figures(figures, filename)

if __name__ == "__main__":
    main()  # Run the main function if this script is executed directly