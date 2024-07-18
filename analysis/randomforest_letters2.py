import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

# Function to create binary features for each letter in Top1_Predicted_Letter
def create_top1_letter_features(df):
    letters = list('abcdefghijklmnopqrstuvwxyz')
    for letter in letters:
        df[f'Top1_{letter}'] = df['Top1_Predicted_Letter'].apply(lambda x: 1 if x == letter else 0)
    return df

# Function to run the analysis for a given dataset, focusing on the importance of individual letters
def run_analysis_excluding_confidence(dataset_path):
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"File not found: {dataset_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"No data: {dataset_path}")
        return None
    except pd.errors.ParserError:
        print(f"Error parsing file: {dataset_path}")
        return None

    df = create_top1_letter_features(df)
    
    # Ensure only binary letter columns are selected, excluding Top1_Confidence
    binary_columns = [col for col in df.columns if col.startswith('Top1_') and col != 'Top1_Confidence' and df[col].dtype in [np.int64, np.float64]]
    
    # Define features and target
    features = df[binary_columns]
    target = df['Top1_Is_Accurate']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    
    # Train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Extract feature importance
    feature_importances = rf_model.feature_importances_
    feature_names = features.columns
    
    # Combine the results
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    
    return importance_df

def aggregate_top_features(top_features_dict, top_letters):
    top_features = {'Letter': top_letters}
    for dataset_name, importance_df in top_features_dict.items():
        if importance_df is not None:
            top_features[dataset_name] = importance_df.set_index('Feature').reindex([f'Top1_{letter}' for letter in top_letters])['Importance'].values
    return pd.DataFrame(top_features)

def plot_feature_importance(melted_df):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=melted_df, x='Letter', y='Importance', hue='Dataset')
    plt.title('Feature Importance of Top Letters Across Datasets')
    plt.xlabel('Letter')
    plt.ylabel('Importance')
    plt.legend(title='Dataset')
    plt.show()

def perform_anova(top_features, letters):
    importance_scores = [top_features.loc[top_features['Letter'] == letter, top_features.columns[1:]].values.flatten() for letter in letters]
    f_statistic, p_value = f_oneway(*importance_scores)
    print(f'ANOVA results for top letters {letters}: F-statistic = {f_statistic}, p-value = {p_value}')
    if p_value < 0.05:
        print("The differences in importance scores for top letters are statistically significant.")
    else:
        print("The differences in importance scores for top letters are not statistically significant.")

# Define dataset paths
dataset_paths = {
    "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv')
}

# Run analysis for each dataset
top_features_dict = {}
combined_df = pd.DataFrame()

for name, path in dataset_paths.items():
    print(f"Running analysis for {name} dataset...")
    importance_df = run_analysis_excluding_confidence(path)
    if importance_df is not None:
        top_features_dict[name] = importance_df
        print(f"Feature importance for {name} dataset:")
        print(importance_df)
        print("\n" + "="*80 + "\n")
        
        # Combine datasets for overall analysis
        df = pd.read_csv(path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

# Define top letters to analyze based on feature importance
top_letters = ['n', 'y', 'd', 't', 'w', 'm', 'u']

# Aggregate top features across all datasets
top_features_df = aggregate_top_features(top_features_dict, top_letters)

# Melt the DataFrame for easier plotting
melted_df = top_features_df.melt(id_vars='Letter', var_name='Dataset', value_name='Importance')

# Plot the feature importance
plot_feature_importance(melted_df)

# Perform ANOVA to compare importance scores for 'n', 'y', and 'd'
perform_anova(top_features_df, ['n', 'y', 'd'])

# Analyze combined dataset
combined_df = create_top1_letter_features(combined_df)

# Ensure only binary letter columns are selected, excluding Top1_Confidence
binary_columns = [col for col in combined_df.columns if col.startswith('Top1_') and col != 'Top1_Confidence' and combined_df[col].dtype in [np.int64, np.float64]]

# Define features and target for combined dataset
features_combined = combined_df[binary_columns]
target_combined = combined_df['Top1_Is_Accurate']

# Split the combined data into training and testing sets
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(features_combined, target_combined, test_size=0.3, random_state=42)

# Train the Random Forest model on the combined dataset
rf_model_combined = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_combined.fit(X_train_combined, y_train_combined)

# Extract feature importance for combined dataset
feature_importances_combined = rf_model_combined.feature_importances_
feature_names_combined = features_combined.columns

# Combine the results for combined dataset
importance_df_combined = pd.DataFrame({'Feature': feature_names_combined, 'Importance': feature_importances_combined})
importance_df_combined.sort_values(by='Importance', ascending=False, inplace=True)

print("Feature importance for combined dataset:")
print(importance_df_combined)

# Plot the feature importance for combined dataset
plt.figure(figsize=(12, 8))
sns.barplot(data=importance_df_combined, x='Feature', y='Importance')
plt.title('Feature Importance of Letters in Combined Dataset')
plt.xlabel('Letter')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.show()
