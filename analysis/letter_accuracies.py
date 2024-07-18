import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
import logging
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Configure logging
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

def load_dataset(args: Tuple[str, Path]) -> Tuple[str, Optional[pd.DataFrame]]:
    """Loads dataset from the given path, handling errors."""
    name, path = args
    try:
        logger.info(f"Loading dataset from {path}")
        df = pd.read_csv(path, usecols=['Top1_Predicted_Letter', 'Top1_Is_Accurate'])
        return name, df
    except Exception as e:
        logger.error(f"Error loading dataset {name} from {path}: {e}")
        return name, None

def create_top1_letter_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates binary features for each letter in Top1_Predicted_Letter."""
    return pd.get_dummies(df['Top1_Predicted_Letter'], prefix='Top1')

def calculate_letter_accuracies(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates accuracy for each letter."""
    accuracies = df.groupby('Top1_Predicted_Letter')['Top1_Is_Accurate'].agg(['mean', 'count', 'std']).reset_index()
    accuracies.columns = ['Letter', 'Accuracy', 'Count', 'Std']
    accuracies['SE'] = accuracies['Std'] / np.sqrt(accuracies['Count'])
    return accuracies.sort_values(by='Accuracy', ascending=False)

def run_accuracy_analysis(args: Tuple[str, Path]) -> Tuple[str, Optional[pd.DataFrame]]:
    """Runs the accuracy analysis for a given dataset."""
    name, path = args
    _, df = load_dataset((name, path))
    if df is not None:
        letter_features = create_top1_letter_features(df)
        df = pd.concat([df, letter_features], axis=1)
        return name, calculate_letter_accuracies(df)
    return name, None

def aggregate_accuracies(accuracy_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregates accuracies from multiple datasets."""
    all_letters = set()
    for df in accuracy_dict.values():
        if df is not None:
            all_letters.update(df['Letter'])
    
    aggregated_data = []
    for letter in all_letters:
        letter_data = {'Letter': letter}
        for dataset, df in accuracy_dict.items():
            if df is not None and letter in df['Letter'].values:
                letter_data[f'{dataset}_Accuracy'] = df[df['Letter'] == letter]['Accuracy'].values[0]
                letter_data[f'{dataset}_Count'] = df[df['Letter'] == letter]['Count'].values[0]
        aggregated_data.append(letter_data)
    
    return pd.DataFrame(aggregated_data)

def plot_accuracies(accuracy_df: pd.DataFrame, title: str, output_path: Path) -> None:
    """Plots accuracies for the letters with improved visualization."""
    plt.figure(figsize=(20, 12))
    
    # Sort the dataframe by accuracy
    accuracy_df = accuracy_df.sort_values('Accuracy', ascending=False)
    
    # Create the bar plot
    ax = sns.barplot(data=accuracy_df, x='Letter', y='Accuracy', hue='Letter', palette='viridis', alpha=0.8, dodge=False)
    
    # Add error bars manually
    ax.errorbar(x=range(len(accuracy_df)), y=accuracy_df['Accuracy'], yerr=accuracy_df['SE'], fmt='none', c='black', capsize=3)
    
    plt.title(title, fontsize=20, fontweight='bold')
    plt.xlabel('Letter', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Annotate each bar with the accuracy value and count
    for i, row in enumerate(accuracy_df.itertuples()):
        ax.text(i, row.Accuracy, f'{row.Accuracy:.3f}\nn={row.Count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add a horizontal line for the mean accuracy
    mean_accuracy = accuracy_df['Accuracy'].mean()
    plt.axhline(y=mean_accuracy, color='r', linestyle='--', alpha=0.7)
    plt.text(len(accuracy_df)-1, mean_accuracy, f'Mean: {mean_accuracy:.3f}', fontsize=12, va='bottom', ha='right', color='r')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_aggregated_accuracies(aggregated_accuracies: pd.DataFrame, output_path: Path) -> None:
    """Plots aggregated accuracies for the letters across datasets."""
    plt.figure(figsize=(20, 12))
    
    # Melt the dataframe to long format for easier plotting
    melted_df = pd.melt(aggregated_accuracies, id_vars=['Letter'], 
                        value_vars=[col for col in aggregated_accuracies.columns if col.endswith('_Accuracy')],
                        var_name='Dataset', value_name='Accuracy')
    
    # Create the bar plot
    ax = sns.barplot(data=melted_df, x='Letter', y='Accuracy', hue='Dataset', palette='viridis', alpha=0.8)
    
    plt.title('Aggregated Accuracy of Letters Across Datasets', fontsize=20, fontweight='bold')
    plt.xlabel('Letter', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def perform_anova_on_accuracies(aggregated_accuracies: pd.DataFrame) -> None:
    """Performs ANOVA on accuracies of letters across datasets."""
    accuracy_columns = [col for col in aggregated_accuracies.columns if col.endswith('_Accuracy')]
    accuracy_data = [aggregated_accuracies[col].dropna() for col in accuracy_columns]
    
    f_statistic, p_value = f_oneway(*accuracy_data)
    logger.info(f'ANOVA results: F-statistic = {f_statistic}, p-value = {p_value}')
    
    if p_value < 0.05:
        logger.info("The differences in accuracy scores across datasets are statistically significant.")
    else:
        logger.info("The differences in accuracy scores across datasets are not statistically significant.")

def main():
    output_dir = Path('output/letter_accuracy')
    output_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor() as executor:
        accuracy_dict = dict(executor.map(run_accuracy_analysis, DATASET_PATHS.items()))

    # Remove None values (failed loads) from the dictionary
    accuracy_dict = {k: v for k, v in accuracy_dict.items() if v is not None}

    for name, accuracy_df in accuracy_dict.items():
        if accuracy_df is not None:
            logger.info(f"Letter accuracy for {name} dataset:\n{accuracy_df}\n" + "="*80)
            plot_accuracies(accuracy_df, f'Accuracy of Letters in {name} Dataset', output_dir / f'{name}_letter_accuracy.png')

    # Aggregate accuracies across datasets
    aggregated_accuracies = aggregate_accuracies(accuracy_dict)
    logger.info("Aggregated letter accuracies across datasets:\n%s", aggregated_accuracies)

    # Perform ANOVA
    perform_anova_on_accuracies(aggregated_accuracies)

if __name__ == "__main__":
    main()
