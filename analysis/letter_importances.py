import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to create binary features for each letter in Top1_Predicted_Letter
def create_top1_letter_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates binary features for each letter in Top1_Predicted_Letter."""
    return pd.get_dummies(df['Top1_Predicted_Letter'], prefix='Top1')

# Function to run the analysis for a given dataset, focusing on the importance of individual letters
def run_analysis_excluding_confidence(dataset_path):
    try:
        df = pd.read_csv(dataset_path)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logger.error(f"Error loading dataset {dataset_path}: {e}")
        return None

    logger.info(f"Columns in dataset: {df.columns.tolist()}")
    required_columns = ['Top1_Predicted_Letter', 'Top1_Is_Accurate']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Missing expected columns in dataset: {dataset_path}")
        return None

    df = create_top1_letter_features(df)
    
    # Ensure only binary letter columns are selected, excluding Top1_Confidence
    binary_columns = [col for col in df.columns if col.startswith('Top1_') and col != 'Top1_Confidence']
    
    # Define features and target
    features = df[binary_columns]
    target = df['Top1_Is_Accurate']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    
    # Train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Extract feature importance
    importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': rf_model.feature_importances_})
    importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    
    return importance_df

def aggregate_top_features(top_features_dict, top_n=7):
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

def plot_feature_importance(df, title, output_path):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='Letter', y='Importance', hue='Dataset')
    plt.title(title)
    plt.xlabel('Letter')
    plt.ylabel('Importance')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def perform_anova(df, letters):
    importance_scores = [df.loc[df['Letter'] == letter, df.columns[1:]].values.flatten() for letter in letters]
    f_statistic, p_value = f_oneway(*importance_scores)
    logger.info(f'ANOVA results for top letters {letters}: F-statistic = {f_statistic}, p-value = {p_value}')
    if p_value < 0.05:
        logger.info("The differences in importance scores for top letters are statistically significant.")
    else:
        logger.info("The differences in importance scores for top letters are not statistically significant.")

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
    combined_df = pd.DataFrame()

    for name, path in dataset_paths.items():
        logger.info(f"Running analysis for {name} dataset...")
        importance_df = run_analysis_excluding_confidence(path)
        if importance_df is not None:
            top_features_dict[name] = importance_df
            logger.info(f"Feature importance for {name} dataset:\n{importance_df.head(10).to_string()}\n" + "="*80)
            
            # Combine datasets for overall analysis
            df = pd.read_csv(path)
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Aggregate top features across all datasets
    if top_features_dict:
        top_features_df = aggregate_top_features(top_features_dict)
        logger.info(f"Aggregated feature importance:\n{top_features_df.to_string()}")

        # Melt the DataFrame for easier plotting
        melted_df = top_features_df.melt(id_vars='Letter', var_name='Dataset', value_name='Importance')

        # Plot the feature importance
        plot_feature_importance(melted_df, 'Feature Importance of Top Letters Across Datasets', output_dir / 'aggregated_feature_importance.png')

        # Perform ANOVA to compare importance scores for top letters
        perform_anova(top_features_df, top_features_df['Letter'].tolist())

    # Analyze combined dataset
    combined_df = create_top1_letter_features(combined_df)

    # Ensure only binary letter columns are selected, excluding Top1_Confidence
    binary_columns = [col for col in combined_df.columns if col.startswith('Top1_') and col != 'Top1_Confidence']

    # Define features and target for combined dataset
    features_combined = combined_df[binary_columns]
    target_combined = combined_df['Top1_Is_Accurate']

    # Split the combined data into training and testing sets
    X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(features_combined, target_combined, test_size=0.3, random_state=42)

    # Train the Random Forest model on the combined dataset
    rf_model_combined = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_combined.fit(X_train_combined, y_train_combined)

    # Extract feature importance for combined dataset
    importance_df_combined = pd.DataFrame({'Feature': features_combined.columns, 'Importance': rf_model_combined.feature_importances_})
    importance_df_combined.sort_values(by='Importance', ascending=False, inplace=True)

    logger.info("Feature importance for combined dataset:")
    logger.info(f"{importance_df_combined.head(10).to_string()}")

    # Plot the feature importance for combined dataset
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df_combined, x='Feature', y='Importance')
    plt.title('Feature Importance of Letters in Combined Dataset')
    plt.xlabel('Letter')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_feature_importance.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
