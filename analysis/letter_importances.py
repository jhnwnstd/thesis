import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Optional

# Configuration
N_ESTIMATORS = 100
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define dataset paths
DATASET_PATHS: Dict[str, Path] = {
    "CLMET3": Path('main/data/outputs/csv/sorted_tokens_clmet_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Lampeter": Path('main/data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Edges": Path('main/data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "CMU": Path('main/data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Brown": Path('main/data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv')
}

def load_dataset(name: str, path: Path) -> Optional[pd.DataFrame]:
    """Load dataset and handle potential errors."""
    try:
        logger.info(f"Loading dataset {name} from {path}")
        df = pd.read_csv(path, usecols=['Top1_Predicted_Letter', 'Top1_Is_Accurate'])
        return df
    except Exception as e:
        logger.error(f"Error loading dataset {name}: {e}")
        return None

def run_analysis(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Run RandomForest analysis and return feature importances."""
    try:
        logger.info(f"Running analysis on dataset: {name}")
        df['Top1_Is_Accurate'] = df['Top1_Is_Accurate'].astype(int)
        
        features = pd.get_dummies(df['Top1_Predicted_Letter'], prefix='Top1')
        target = df['Top1_Is_Accurate']
        
        logger.info("Training RandomForestClassifier")
        rf_model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
        rf_model.fit(features, target)
        
        importance_df = pd.DataFrame({
            'Feature': features.columns.str.replace('Top1_', ''),
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        logger.info(f"Feature importance analysis complete for dataset: {name}")
        return importance_df
    except Exception as e:
        logger.error(f"Error processing {name}: {e}")
        return pd.DataFrame()

def plot_feature_importances(df: pd.DataFrame, name: str, output_dir: Path):
    """Plot and save feature importances."""
    logger.info(f"Plotting feature importances for {name}")
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='Feature', y='Importance')
    plt.title(f'Feature Importance of Top Letters in {name} Dataset', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / f'{name}_feature_importance.png', dpi=300)
    plt.close()
    logger.info(f"Plotting complete for {name}")

def main():
    logger.info("Starting main process")
    output_dir = Path('output/feature_importance')
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, path in DATASET_PATHS.items():
        df = load_dataset(name, path)
        if df is not None:
            importance_df = run_analysis(name, df)
            if not importance_df.empty:
                plot_feature_importances(importance_df, name, output_dir)
                logger.info(f"Completed analysis for {name}")

    logger.info("Main process complete")

if __name__ == "__main__":
    main()