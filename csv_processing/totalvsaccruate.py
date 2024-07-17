import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Define dataset paths
datasets = {
    "CLMET3": Path('main/data/outputs/csv/sorted_tokens_clmet_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Lampeter": Path('main/data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Edges": Path('main/data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "CMU": Path('main/data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Brown": Path('main/data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv')
}

def plot_predictions(data: pd.DataFrame, dataset_name: str) -> None:
    """Plot the distribution of predicted letters and their accuracies."""
    plt.figure(figsize=(12, 6))
    predicted_letter_counts = data['Top1_Predicted_Letter'].value_counts().sort_index()
    plt.bar(predicted_letter_counts.index, predicted_letter_counts.values, label='Total Predictions', alpha=0.7)
    
    accurate_predicted_letter_counts = data[data['Top1_Is_Accurate']]['Top1_Predicted_Letter'].value_counts().sort_index()
    plt.bar(accurate_predicted_letter_counts.index, accurate_predicted_letter_counts.values, label='Accurate Predictions', alpha=0.7)
    
    plt.title(f'Distribution of Predicted Letters and Their Accuracies ({dataset_name})')
    plt.xlabel('Predicted Letter')
    plt.ylabel('Number of Predictions')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Plot for each individual dataset
for name, path in datasets.items():
    data = pd.read_csv(path)
    plot_predictions(data, name)

# Combine all datasets into one DataFrame
combined_data = pd.concat([pd.read_csv(path) for path in datasets.values()], ignore_index=True)

# Plot for the combined dataset
plot_predictions(combined_data, 'Combined Datasets')
