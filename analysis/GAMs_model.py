import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pygam import LogisticGAM, s
import logging
from typing import Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths to datasets
DATASET_PATHS = {
    "CLMET3": Path('main/data/outputs/csv/sorted_tokens_clmet_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Lampeter": Path('main/data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Edges": Path('main/data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "CMU": Path('main/data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Brown": Path('main/data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv')
}

def load_data(filepath: Path) -> Optional[pd.DataFrame]:
    """Load data from a CSV file."""
    try:
        # Attempt to read the CSV file
        data = pd.read_csv(filepath)
        # Log success message
        logger.info(f"Data loaded successfully from {filepath}")
        # Return the loaded data
        return data
    except FileNotFoundError:
        # Log error message if file is not found
        logger.error(f"File not found: {filepath}")
        # Return None to indicate failure
        return None

def prepare_data(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Prepare data by calculating word length and normalized missing letter index."""
    
    # Define the columns that are required for the analysis
    required_columns = {'Top1_Is_Accurate', 'Tested_Word'}
    
    # Check if the required columns are present in the dataframe
    if not required_columns.issubset(data.columns):
        logger.error("Required columns are missing")
        return None  # Return None if required columns are missing

    # Calculate the length of each word in the 'Tested_Word' column
    # This is used later for normalizing the missing letter index
    data['Word_Length'] = data['Tested_Word'].str.len()

    # Calculate the normalized index of the missing letter
    # 1. Find the index of '_' in each word
    # 2. Divide by (word length - 1) to normalize
    # This gives a value between 0 and 1, representing the relative position of the missing letter
    data['Normalized_Missing_Index'] = data['Tested_Word'].str.find('_') / (data['Word_Length'] - 1)

    # Clean up the data:
    # 1. Replace infinity values with NaN
    #    (This can happen if the word length is 1, causing division by zero)
    # 2. Drop all rows with NaN values
    # This ensures we only keep valid, computable data points
    data = data.replace({'Normalized_Missing_Index': {np.inf: np.nan, -np.inf: np.nan}}).dropna()
    
    return data  # Return the cleaned and prepared dataframe

# Explanation of the function's design:
# 1. Input validation: The function first checks for required columns. This prevents
#    errors later in the processing and provides clear feedback on data issues.
#
# 2. Word length calculation: This is done separately as it's used in the next step
#    and might be useful for other analyses.
#
# 3. Normalized missing index: This is the key feature for the analysis. It represents
#    the position of the missing letter, normalized by word length. This allows
#    comparison across words of different lengths.
#
# 4. Data cleaning: Replacing infinities with NaN and then dropping NaNs ensures
#    that only valid data points are kept. This is crucial for accurate analysis.
#
# 5. The function returns None for invalid input, allowing the calling code to
#    handle this case appropriately.
#
# This design ensures that the output data is clean, normalized, and ready for
# further analysis or model fitting.

def fit_model(X: pd.DataFrame, y: pd.Series, n_splines: int = 15) -> Optional[LogisticGAM]:
    """Fit a logistic GAM model."""
    try:
        # Create and fit the GAM model
        gam = LogisticGAM(s(0, n_splines=n_splines)).fit(X, y)
        logger.info("Model fitting complete")
        return gam
    except Exception as e:
        # Log any errors that occur during model fitting
        logger.error(f"Error fitting model: {str(e)}")
        return None

def plot_results(XX: np.ndarray, proba: np.ndarray, X: np.ndarray, y: np.ndarray, title: str, config: dict, output_path: Path):
    """Plot the results of the GAM model predictions against the actual data."""
    # Create a new figure with specified size
    plt.figure(figsize=config['figsize'])
    # Set the plot style
    sns.set_style("whitegrid")
    # Plot the model prediction line
    plt.plot(XX, proba, label='Model Prediction', color=config['prediction_color'], linewidth=2)
    # Scatter plot the actual data points
    plt.scatter(X, y, color=config['data_color'], alpha=0.7, label='Actual Data')
    # Set labels and title
    plt.xlabel('Normalized Missing Index', fontsize=12)
    plt.ylabel('Prediction Accuracy', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    # Set x-axis ticks
    plt.xticks(np.arange(0, 1.1, 0.1), labels=[f"{tick:.1f}" for tick in np.arange(0, 1.1, 0.1)])
    
    # Adjust y-axis range if dynamic_range is True
    if config['dynamic_range']:
        center_point = np.median(proba)
        margin = 0.30
        plt.ylim([max(0, center_point - margin), min(1, center_point + margin)])
    
    # Adjust layout and save the plot
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate various performance metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }

def process_dataset(args: Tuple[str, Path, dict]) -> Optional[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]]:
    """Process each dataset: load data, prepare it, fit the model, plot results, and calculate metrics."""
    # Unpack arguments
    name, path, config = args
    # Load the data
    data = load_data(path)
    if data is None:
        return None

    # Prepare the data
    prepared_data = prepare_data(data)
    if prepared_data is None:
        return None

    # Extract features and target
    X = prepared_data[['Normalized_Missing_Index']]
    y = prepared_data['Top1_Is_Accurate']
    # Fit the GAM model
    gam = fit_model(X, y)
    if gam is None:
        return None

    # Generate points for plotting the model prediction
    XX = np.linspace(0, 1, 1000)[:, None]
    proba = gam.predict_proba(XX)
    # Plot and save results
    output_path = Path('output/gams') / f"{name}_GAM_df.png"
    plot_results(XX.ravel(), proba, X.to_numpy().ravel(), y, f'Effect of Normalized Missing Index on Prediction Accuracy in {name}', config, output_path)

    # Calculate performance metrics
    y_pred = gam.predict(X) > 0.5
    metrics = calculate_metrics(y, y_pred)

    return name, X.to_numpy().ravel(), y, XX.ravel(), proba, metrics

def plot_all_datasets(datasets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], config: dict, output_path: Path):
    """Plot the results of all datasets on a single graph."""
    # Create a new figure with specified size
    plt.figure(figsize=config['figsize'])
    # Set the plot style
    sns.set_style("whitegrid")
    
    # Plot each dataset
    for name, (X, y, XX, proba) in datasets.items():
        plt.plot(XX, proba, label=f'{name} Model Prediction', linewidth=2)
        plt.scatter(X, y, alpha=0.7, label=f'{name} Actual Data')
    
    # Set labels and title
    plt.xlabel('Normalized Missing Index', fontsize=12)
    plt.ylabel('Prediction Accuracy', fontsize=12)
    plt.title('Effect of Normalized Missing Index on Prediction Accuracy across Datasets', fontsize=14)
    plt.legend()
    # Set x-axis ticks
    plt.xticks(np.arange(0, 1.1, 0.1), labels=[f"{tick:.1f}" for tick in np.arange(0, 1.1, 0.1)])
    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_combined_dataset(datasets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], config: dict) -> Optional[Dict[str, float]]:
    """Process all datasets, combine them, fit a single model, plot the results, and return metrics."""
    # Combine X and y from all datasets
    combined_X = np.concatenate([X for X, _, _, _ in datasets.values()])
    combined_y = np.concatenate([y for _, y, _, _ in datasets.values()])
    
    # Fit the GAM model on combined data
    gam = fit_model(combined_X.reshape(-1, 1), combined_y)
    if gam is None:
        return None

    # Generate points for plotting the model prediction
    XX = np.linspace(0, 1, 1000)[:, None]
    proba = gam.predict_proba(XX)
    # Plot and save results
    output_path = Path('output/gams/combined_dataset_GAM_df.png')
    plot_results(XX.ravel(), proba, combined_X, combined_y, 'Effect of Normalized Missing Index on Prediction Accuracy for Combined Dataset', config, output_path)

    # Calculate performance metrics
    y_pred = gam.predict(combined_X.reshape(-1, 1)) > 0.5
    return calculate_metrics(combined_y, y_pred)

def main():
    # Define default plot configuration
    default_plot_config = {
        'figsize': (14, 8),
        'prediction_color': 'blue',
        'data_color': 'black',
        'dynamic_range': True
    }

    # Process individual datasets in parallel
    with ProcessPoolExecutor() as executor:
        # Submit tasks for each dataset
        futures = [executor.submit(process_dataset, (name, path, default_plot_config)) for name, path in DATASET_PATHS.items()]
        datasets = {}
        all_metrics = {}
        # Collect results as they complete
        for future in as_completed(futures):
            result = future.result()
            if result:
                name, X, y, XX, proba, metrics = result
                datasets[name] = (X, y, XX, proba)
                all_metrics[name] = metrics

    # Process and plot all datasets together
    plot_all_datasets(datasets, default_plot_config, Path('output/gams/all_datasets_GAM_df.png'))

    # Process and plot the combined dataset
    combined_metrics = process_combined_dataset(datasets, default_plot_config)

    # Print metrics for all datasets and the combined dataset
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