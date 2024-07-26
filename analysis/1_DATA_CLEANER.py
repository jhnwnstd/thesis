import pandas as pd
from pathlib import Path
import logging
from scipy.stats import zscore

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def standardize_text_columns(df, columns):
    df[columns] = df[columns].apply(lambda x: x.str.lower().str.strip())
    return df

def drop_null_values(df, column):
    initial_count = len(df)
    df.dropna(subset=[column], inplace=True)
    logger.info(f"Dropped rows with null '{column}'. New row count: {len(df)} (dropped {initial_count - len(df)} rows)")
    return df

def convert_columns(df, columns, dtype):
    df[columns] = df[columns].astype(dtype)
    return df

def remove_duplicates(df):
    initial_row_count = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = initial_row_count - len(df)
    logger.info(f"Removed {duplicates_removed} duplicate rows. New row count: {len(df)}")
    return df

def validate_predicted_letters(df, columns):
    for col in columns:
        df = df[df[col].apply(lambda x: isinstance(x, str) and len(x) == 1)]
    return df

def remove_outliers_conservatively(df, columns, z_threshold=3):
    z_scores = df[columns].apply(zscore)
    df_cleaned = df[(z_scores < z_threshold).all(axis=1)]
    return df_cleaned

def clean_dataframe(df):
    df.dropna(inplace=True)
    confidence_columns = [col for col in df.columns if 'Confidence' in col]
    df = convert_columns(df, confidence_columns, float)
    df = remove_duplicates(df)
    
    # Ensure confidence values are within [0, 1]
    for col in confidence_columns:
        df = df[(df[col] >= 0) & (df[col] <= 1)]

    df = remove_outliers_conservatively(df, confidence_columns)
    return df

def clean_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Read dataset with {len(df)} rows from {file_path.name}")

        df = standardize_text_columns(df, ['Tested_Word', 'Original_Word'])
        df = drop_null_values(df, 'Original_Word')
        boolean_columns = ['In_Training_Set'] + [f'Top{i}_Is_Valid' for i in range(1, 4)] + [f'Top{i}_Is_Accurate' for i in range(1, 4)]
        df = convert_columns(df, boolean_columns, bool)
        df = convert_columns(df, ['Correct_Letter_Rank'], int)
        df = remove_duplicates(df)
        predicted_letter_columns = [f'Top{i}_Predicted_Letter' for i in range(1, 3)]
        df = validate_predicted_letters(df, predicted_letter_columns)
        df = clean_dataframe(df)
        
        df.to_csv(file_path, index=False)
        logger.info(f"Cleaned dataset overwritten at {file_path}.")
        
    except pd.errors.EmptyDataError:
        logger.error(f"{file_path.name} is empty.")
    except pd.errors.ParserError:
        logger.error(f"{file_path.name} could not be parsed.")
    except KeyError as e:
        logger.error(f"Missing expected column in {file_path.name}: {e}")
    except ValueError as e:
        logger.error(f"Data type conversion error in {file_path.name}: {e}")
    except Exception as e:
        logger.error(f"Failed to clean {file_path.name} due to {e}")

def main():
    data_dir = Path('main/data/outputs/csv')
    files_to_clean = list(data_dir.glob('*.csv'))

    for file_path in files_to_clean:
        logger.info(f"Cleaning dataset: {file_path.name}")
        clean_dataset(file_path)

if __name__ == "__main__":
    main()