from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import LogitResults
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dictionary mapping dataset names to their file paths
DATASET_PATHS: Dict[str, Path] = {
    "CLMET3": Path('main/data/outputs/csv/sorted_tokens_clmet_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Lampeter": Path('main/data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Edges": Path('main/data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "CMU": Path('main/data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Brown": Path('main/data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv')
}

def load_and_prepare_data(file_path: Path) -> Optional[pd.DataFrame]:
    """Load and prepare data for analysis."""
    try:
        data = pd.read_csv(file_path, usecols=['Original_Word', 'Top1_Is_Accurate'])
        logger.info(f"Data loaded successfully from {file_path}")
        
        data['Word_Length'] = data['Original_Word'].fillna('').str.len()
        data['Top1_Is_Accurate'] = pd.to_numeric(data['Top1_Is_Accurate'], errors='coerce')
        data = data.dropna(subset=['Top1_Is_Accurate', 'Word_Length'])
        data['Top1_Is_Accurate'] = data['Top1_Is_Accurate'].astype(int)
        
        return data
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return None

def calculate_word_length_stats(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate statistical measures for word length and accuracy."""
    stats = data.groupby('Word_Length').agg({
        'Top1_Is_Accurate': ['mean', 'std', 'count', 'sem']
    }).reset_index()
    stats.columns = ['Word_Length', 'Accuracy_Mean', 'Accuracy_Std', 'Sample_Count', 'Accuracy_SEM']
    return stats

def perform_logistic_regression(data: pd.DataFrame) -> Optional[LogitResults]:
    """Perform logistic regression analysis."""
    try:
        return smf.logit('Top1_Is_Accurate ~ Word_Length', data=data).fit()
    except Exception as e:
        logger.error(f"Error in logistic regression: {str(e)}")
        return None

def plot_accuracy_vs_word_length(stats: pd.DataFrame, dataset_name: str):
    """Create a plot of accuracy vs word length."""
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=stats, x='Word_Length', y='Accuracy_Mean', size='Sample_Count', legend=False)
    
    # Filter out rows with NaN in Accuracy_SEM
    valid_stats = stats.dropna(subset=['Accuracy_SEM'])
    
    plt.errorbar(valid_stats['Word_Length'], valid_stats['Accuracy_Mean'], 
                 yerr=valid_stats['Accuracy_SEM'], fmt='none', alpha=0.5)
    plt.title(f'Accuracy vs Word Length for {dataset_name} Dataset')
    plt.xlabel('Word Length')
    plt.ylabel('Accuracy')
    plt.savefig(f'{dataset_name}_accuracy_vs_word_length.png')
    plt.close()

def analyze_dataset(args: Tuple[str, Path]) -> Dict[str, Optional[Dict]]:
    """Analyze a single dataset."""
    dataset_name, file_path = args
    data = load_and_prepare_data(file_path)
    if data is None:
        return {dataset_name: None}

    stats = calculate_word_length_stats(data)
    model = perform_logistic_regression(data)
    plot_accuracy_vs_word_length(stats, dataset_name)

    return {dataset_name: {
        'stats': stats,
        'model': model,
        'data_types': data.dtypes
    }}

def main():
    results = {}
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(analyze_dataset, (name, path)) for name, path in DATASET_PATHS.items()]
        for future in as_completed(futures):
            results.update(future.result())

    for dataset_name, result in results.items():
        if result is not None:
            logger.info(f"\n{dataset_name} Dataset Analysis:")
            logger.info("Statistical Analysis of Word Length and Prediction Accuracy:")
            logger.info(f"\n{result['stats']}")
            logger.info("\nData Types:")
            logger.info(f"{result['data_types']}")
            if result['model'] is not None:
                logger.info("\nLogistic Regression Analysis Summary:")
                logger.info(f"\n{result['model'].summary()}")
            else:
                logger.info("\nLogistic Regression Analysis failed.")

if __name__ == "__main__":
    main()