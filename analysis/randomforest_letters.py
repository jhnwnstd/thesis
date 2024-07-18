import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Function to create binary features for each letter in Top1_Predicted_Letter
def create_top1_letter_features(df):
    letters = list('abcdefghijklmnopqrstuvwxyz')
    for letter in letters:
        df[f'Top1_{letter}'] = df['Top1_Predicted_Letter'].apply(lambda x: 1 if x == letter else 0)
    return df

# Function to run the analysis for a given dataset
def run_analysis(dataset_path):
    df = pd.read_csv(dataset_path)
    df = create_top1_letter_features(df)
    
    # Ensure only binary letter columns are selected
    binary_columns = [col for col in df.columns if col.startswith('Top1_') and df[col].dtype in [np.int64, np.float64]]
    
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
    importance_data = {'Feature': feature_names, 'Importance': feature_importances}
    importance_df = pd.DataFrame(importance_data)
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    return importance_df

# Define dataset paths
dataset_paths = {
    "CLMET3": Path('data/outputs/csv/CLMET3_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Lampeter": Path('data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Edges": Path('data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "CMU": Path('data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Brown": Path('data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv')
}

# Run analysis for each dataset
for name, path in dataset_paths.items():
    print(f"Running analysis for {name} dataset...")
    importance_df = run_analysis(path)
    print(f"Feature importance for {name} dataset:")
    print(importance_df)
    print("\n" + "="*80 + "\n")