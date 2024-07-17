from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
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
        data = pd.read_csv(file_path, usecols=['Tested_Word', 'Top1_Is_Accurate'])
        logger.info(f"Data loaded successfully from {file_path}")
        
        data['Word_Length'] = data['Tested_Word'].str.len()
        data['Normalized_Missing_Index'] = data['Tested_Word'].str.find('_') / (data['Word_Length'] - 1)
        data['Top1_Is_Accurate'] = pd.to_numeric(data['Top1_Is_Accurate'], errors='coerce')
        
        # Handle infinity and NaN values
        data.replace({'Normalized_Missing_Index': {np.inf: np.nan, -np.inf: np.nan}}, inplace=True)
        data.dropna(subset=['Top1_Is_Accurate', 'Normalized_Missing_Index'], inplace=True)
        data['Top1_Is_Accurate'] = data['Top1_Is_Accurate'].astype(int)
        
        return data
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return None

def calculate_missing_index_stats(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate statistical measures for normalized missing index and accuracy."""
    stats = data.groupby('Normalized_Missing_Index').agg({
        'Top1_Is_Accurate': ['mean', 'std', 'count', 'sem']
    }).reset_index()
    stats.columns = ['Normalized_Missing_Index', 'Accuracy_Mean', 'Accuracy_Std', 'Sample_Count', 'Accuracy_SEM']
    return stats

def perform_logistic_regression(data: pd.DataFrame) -> Optional[smf.logit]:
    """Perform logistic regression analysis."""
    try:
        return smf.logit('Top1_Is_Accurate ~ Normalized_Missing_Index', data=data).fit()
    except Exception as e:
        logger.error(f"Error in logistic regression: {str(e)}")
        return None

def perform_linear_regression(data: pd.DataFrame) -> Optional[smf.ols]:
    """Perform linear regression analysis."""
    try:
        return smf.ols('Top1_Is_Accurate ~ Normalized_Missing_Index', data=data).fit()
    except Exception as e:
        logger.error(f"Error in linear regression: {str(e)}")
        return None

def plot_accuracy_vs_index(stats: pd.DataFrame, dataset_name: str):
    """Create a plot of accuracy vs normalized missing index."""
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=stats, x='Normalized_Missing_Index', y='Accuracy_Mean', size='Sample_Count', legend=False)
    
    # Filter out rows with NaN in Accuracy_SEM
    valid_stats = stats.dropna(subset=['Accuracy_SEM'])
    
    plt.errorbar(valid_stats['Normalized_Missing_Index'], valid_stats['Accuracy_Mean'], 
                 yerr=valid_stats['Accuracy_SEM'], fmt='none', alpha=0.5)
    
    # Add trend line using linear regression
    sns.regplot(x='Normalized_Missing_Index', y='Accuracy_Mean', data=stats, scatter=False, color='red')
    
    plt.title(f'Accuracy vs Normalized Missing Index for {dataset_name} Dataset')
    plt.xlabel('Normalized Missing Index')
    plt.ylabel('Accuracy')
    plt.savefig(f'{dataset_name}_accuracy_vs_index.png')
    plt.show()

def analyze_dataset(args: Tuple[str, Path]) -> Dict[str, Optional[Dict]]:
    """Analyze a single dataset."""
    dataset_name, file_path = args
    data = load_and_prepare_data(file_path)
    if data is None:
        return {dataset_name: None}

    stats = calculate_missing_index_stats(data)
    logit_model = perform_logistic_regression(data)
    ols_model = perform_linear_regression(data)
    plot_accuracy_vs_index(stats, dataset_name)

    return {dataset_name: {
        'stats': stats,
        'logit_model': logit_model,
        'ols_model': ols_model,
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
            logger.info("Statistical Analysis of Normalized Missing Index and Prediction Accuracy:")
            logger.info(f"\n{result['stats']}")
            logger.info("\nData Types:")
            logger.info(f"{result['data_types']}")
            if result['logit_model'] is not None:
                logger.info("\nLogistic Regression Analysis Summary:")
                logger.info(f"\n{result['logit_model'].summary()}")
            else:
                logger.info("\nLogistic Regression Analysis failed.")
            if result['ols_model'] is not None:
                logger.info("\nLinear Regression Analysis Summary:")
                logger.info(f"\n{result['ols_model'].summary()}")
            else:
                logger.info("\nLinear Regression Analysis failed.")

if __name__ == "__main__":
    main()
