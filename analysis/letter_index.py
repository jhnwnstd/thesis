from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import logging
from typing import Dict, Optional

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

def analyze_dataset(dataset_name: str, file_path: Path):
    """Analyze a single dataset."""
    logger.info(f"\nProcessing dataset: {dataset_name}")
    logger.info("=" * 50)
    data = load_and_prepare_data(file_path)
    if data is None:
        return

    stats = calculate_missing_index_stats(data)
    logit_model = perform_logistic_regression(data)
    ols_model = perform_linear_regression(data)

    logger.info(f"\n{dataset_name} Dataset Analysis:")
    logger.info("=" * 50)
    logger.info("Statistical Analysis of Normalized Missing Index and Prediction Accuracy:")
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