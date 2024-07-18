import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Optional

# Configuration
N_ESTIMATORS = 50
RANDOM_STATE = 42
CV_FOLDS = 5

logging.basicConfig(level=logging.INFO, format='%(message)s')
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
    try:
        logger.info(f"Loading dataset {name} from {path}")
        df = pd.read_csv(path, usecols=['Top1_Predicted_Letter', 'Top1_Is_Accurate'])
        return df
    except Exception as e:
        logger.error(f"Error loading dataset {name}: {e}")
        return None

def run_analysis(name: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
    try:
        logger.info(f"Running analysis on dataset: {name}")
        df['Top1_Is_Accurate'] = df['Top1_Is_Accurate'].astype(int)
        
        features = pd.get_dummies(df['Top1_Predicted_Letter'], prefix='Top1')
        target = df['Top1_Is_Accurate']
        
        logger.info("Training RandomForestClassifier with RFECV")
        rf_model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
        rfecv = RFECV(estimator=rf_model, step=1, cv=CV_FOLDS, scoring='accuracy', n_jobs=-1)
        rfecv.fit(features, target)
        
        importance_df = pd.DataFrame({
            'Feature': features.columns[rfecv.support_].str.replace('Top1_', ''),
            'Importance': rfecv.estimator_.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        logger.info(f"Feature importance analysis complete for dataset: {name}")
        return importance_df
    except Exception as e:
        logger.error(f"Error processing {name}: {e}")
        return None

def plot_feature_importances(df: pd.DataFrame, name: str, output_dir: Path):
    logger.info(f"Plotting feature importances for {name}")

    # Ensure all unique letters in the dataset are included, even if their importance is zero
    all_letters = set(df['Feature'])
    missing_letters = all_letters - set(df['Feature'])
    
    if missing_letters:
        missing_df = pd.DataFrame({
            'Feature': list(missing_letters),
            'Importance': [0] * len(missing_letters)
        })
        df = pd.concat([df, missing_df], ignore_index=True)

    # Sort the DataFrame by importance
    df = df.sort_values(by='Importance', ascending=False)

    # Plotting
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
            if importance_df is not None:
                plot_feature_importances(importance_df, name, output_dir)
                logger.info(f"Completed analysis for {name}")

    logger.info("Main process complete")

if __name__ == "__main__":
    main()