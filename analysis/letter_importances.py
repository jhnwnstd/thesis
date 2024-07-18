import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Configuration
N_ESTIMATORS = 100
RANDOM_STATE = 42
TOP_N_FEATURES = 7
CV_FOLDS = 5

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def create_top1_letter_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates binary features for each letter in Top1_Predicted_Letter while preserving other columns."""
    letter_features = pd.get_dummies(df['Top1_Predicted_Letter'], prefix='Top1')
    return pd.concat([df.drop('Top1_Predicted_Letter', axis=1), letter_features], axis=1)

def run_analysis_excluding_confidence(dataset_path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(dataset_path, usecols=['Top1_Predicted_Letter', 'Top1_Is_Accurate'])
        df['Top1_Is_Accurate'] = df['Top1_Is_Accurate'].astype(int)
        
        df = create_top1_letter_features(df)
        
        binary_columns = [col for col in df.columns if col.startswith('Top1_') and col != 'Top1_Is_Accurate']
        features = df[binary_columns]
        target = df['Top1_Is_Accurate']
        
        rf_model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
        
        # Perform Recursive Feature Elimination with Cross-Validation
        rfecv = RFECV(estimator=rf_model, step=1, cv=CV_FOLDS, scoring='accuracy', n_jobs=-1)
        rfecv.fit(features, target)
        
        # Get feature importance scores
        importance = rfecv.estimator_.feature_importances_
        feature_names = features.columns[rfecv.support_]
        
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        return importance_df.sort_values(by='Importance', ascending=False)
    
    except Exception as e:
        logger.error(f"Error processing {dataset_path}: {str(e)}")
        logger.exception("Exception details:")
        return None

def aggregate_top_features(top_features_dict: Dict[str, pd.DataFrame], top_n: int = TOP_N_FEATURES) -> pd.DataFrame:
    all_importances = pd.concat(top_features_dict.values())
    top_letters = all_importances.groupby('Feature')['Importance'].mean().nlargest(top_n).index.str.replace('Top1_', '')
    
    aggregated_data = []
    for letter in top_letters:
        letter_data = {'Letter': letter}
        for dataset, df in top_features_dict.items():
            if df is not None:
                letter_data[dataset] = df.loc[df['Feature'] == f'Top1_{letter}', 'Importance'].values[0] if f'Top1_{letter}' in df['Feature'].values else np.nan
        aggregated_data.append(letter_data)
    return pd.DataFrame(aggregated_data)

def plot_feature_importance(df: pd.DataFrame, title: str, output_path: Path):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='Letter', y='Importance', hue='Dataset')
    plt.title(title, fontsize=16)
    plt.xlabel('Letter', fontsize=14)
    plt.ylabel('Importance', fontsize=14)
    plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_heatmap(df: pd.DataFrame, output_path: Path):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.set_index('Letter'), annot=True, cmap='YlOrRd', fmt='.3f')
    plt.title('Feature Importance Heatmap Across Datasets', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def perform_statistical_tests(df: pd.DataFrame):
    # Prepare data for tests
    letter_importances = {letter: df.loc[df['Letter'] == letter, df.columns[1:]].values.flatten() for letter in df['Letter']}
    letter_importances = {k: v[~np.isnan(v)] for k, v in letter_importances.items()}
    
    # ANOVA
    f_statistic, p_value = stats.f_oneway(*letter_importances.values())
    logger.info(f'ANOVA results: F-statistic = {f_statistic:.4f}, p-value = {p_value:.4f}')
    
    # Kruskal-Wallis H-test (non-parametric alternative to ANOVA)
    h_statistic, p_value = stats.kruskal(*letter_importances.values())
    logger.info(f'Kruskal-Wallis H-test results: H-statistic = {h_statistic:.4f}, p-value = {p_value:.4f}')
    
    if p_value < 0.05:
        logger.info("The differences in importance scores for letters are statistically significant.")
        
        # Post-hoc pairwise t-tests
        from itertools import combinations
        for l1, l2 in combinations(letter_importances.keys(), 2):
            t_stat, p_val = stats.ttest_ind(letter_importances[l1], letter_importances[l2])
            logger.info(f"Pairwise t-test {l1} vs {l2}: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")
    else:
        logger.info("The differences in importance scores for letters are not statistically significant.")

def process_dataset(args: Tuple[str, Path]) -> Tuple[str, Optional[pd.DataFrame]]:
    name, path = args
    logger.info(f"Running analysis for {name} dataset...")
    return name, run_analysis_excluding_confidence(path)

def main():
    output_dir = Path('output/feature_importance')
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_paths = {
        "CLMET3": Path('main/data/outputs/csv/sorted_tokens_clmet_context_sensitive_split0.5_qrange7-7_prediction.csv'),
        "Lampeter": Path('main/data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv'),
        "Edges": Path('main/data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv'),
        "CMU": Path('main/data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv'),
        "Brown": Path('main/data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv')
    }

    top_features_dict = {}
    
    # Process datasets in parallel
    with ProcessPoolExecutor() as executor:
        future_to_name = {executor.submit(process_dataset, (name, path)): name for name, path in dataset_paths.items()}
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                _, importance_df = future.result()
                if importance_df is not None:
                    top_features_dict[name] = importance_df
                    logger.info(f"Feature importance for {name} dataset:\n{importance_df.head(10).to_string()}\n" + "="*80)
            except Exception as exc:
                logger.error(f'{name} generated an exception: {exc}')

    # Aggregate top features across all datasets
    if top_features_dict:
        top_features_df = aggregate_top_features(top_features_dict)
        logger.info(f"Aggregated feature importance:\n{top_features_df.to_string()}")

        melted_df = top_features_df.melt(id_vars='Letter', var_name='Dataset', value_name='Importance')
        plot_feature_importance(melted_df, 'Feature Importance of Top Letters Across Datasets', output_dir / 'aggregated_feature_importance.png')
        plot_heatmap(top_features_df, output_dir / 'feature_importance_heatmap.png')
        perform_statistical_tests(top_features_df)

        # Additional analysis
        logger.info("\nTop 3 most important letters for each dataset:")
        for dataset in top_features_df.columns[1:]:
            top_3 = top_features_df.nlargest(3, dataset)[['Letter', dataset]]
            logger.info(f"{dataset}:\n{top_3.to_string(index=False)}\n")

        logger.info("\nOverall top 3 letters across all datasets:")
        overall_top_3 = top_features_df.set_index('Letter').mean(axis=1).nlargest(3)
        logger.info(overall_top_3.to_string())

    # Explicitly clean up any resources
    import gc
    gc.collect()

if __name__ == "__main__":
    main()