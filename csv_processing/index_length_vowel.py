import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, log_loss
from pathlib import Path

def is_vowel(letter: str) -> bool:
    """Check if a letter is a vowel."""
    return letter.lower() in 'aeiouy'

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data for model training."""
    data['Word_Length'] = data['Original_Word'].apply(len)
    data['Missing_Letter_Position'] = data['Tested_Word'].str.find('_')
    data['Relative_Position'] = data['Missing_Letter_Position'] / data['Word_Length']
    data['Missing_Letter'] = data.apply(lambda row: row['Original_Word'][row['Missing_Letter_Position']] if row['Missing_Letter_Position'] != -1 else '', axis=1)
    data['Is_Vowel'] = data['Missing_Letter'].apply(is_vowel).astype(int)
    data['Top1_Is_Accurate'] = data['Top1_Is_Accurate'].astype(int)
    return data

datasets = {
    "CLMET3": Path('main/data/outputs/csv/sorted_tokens_clmet_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Lampeter": Path('main/data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Edges": Path('main/data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "CMU": Path('main/data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Brown": Path('main/data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv')
}

def logistic_regression_analysis(dataset_name: str, file_path: Path) -> None:
    """Run logistic regression analysis on the dataset."""
    data = pd.read_csv(file_path)
    data = preprocess_data(data)
    
    X = data[['Relative_Position', 'Word_Length', 'Is_Vowel']]
    y = data['Top1_Is_Accurate']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]
    
    log_likelihood_model = -log_loss(y_test, y_pred_proba, normalize=False)
    log_likelihood_null = -log_loss(y_test, np.ones_like(y_test) * y_test.mean(), normalize=False)
    
    pseudo_r_squared = 1 - (log_likelihood_model / log_likelihood_null)
    
    print(f'\n{dataset_name} Dataset Analysis:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Pseudo R-squared: {pseudo_r_squared:.4f}')
    print('Classification Report:')
    print(report)
    print('Model Coefficients:')
    print(f'  Relative Position: {coefficients[0]:.4f}')
    print(f'  Word Length: {coefficients[1]:.4f}')
    print(f'  Is Vowel: {coefficients[2]:.4f}')
    print(f'  Intercept: {intercept:.4f}')

for dataset_name, file_path in datasets.items():
    logistic_regression_analysis(dataset_name, file_path)
