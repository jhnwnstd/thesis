import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def standardize_text_columns(df, columns):
    for col in columns:
        df[col] = df[col].str.lower().str.strip()
    return df

def drop_null_values(df, column):
    initial_count = len(df)
    df.dropna(subset=[column], inplace=True)
    logger.info(f"Dropped rows with null '{column}'. New row count: {len(df)} (dropped {initial_count - len(df)} rows)")
    return df

def convert_to_boolean(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df

def convert_to_integer(df, column):
    if column in df.columns:
        df[column] = df[column].astype(int)
    return df

def remove_duplicates(df):
    initial_row_count = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = initial_row_count - len(df)
    logger.info(f"Removed {duplicates_removed} duplicate rows. New row count: {len(df)}")
    return df

def validate_predicted_letters(df, columns):
    for col in columns:
        if col in df.columns:
            df = df[df[col].apply(lambda x: isinstance(x, str) and len(x) == 1)]
    return df

def clean_dataframe(df):
    # Step 1: Identify and Handle Missing Values
    df = df.dropna()

    # Step 2: Ensure Correct Data Types
    confidence_columns = [col for col in df.columns if 'Confidence' in col]
    for col in confidence_columns:
        df[col] = df[col].astype(float)

    # Step 3: Remove Duplicate Rows
    df = df.drop_duplicates()

    # Step 4: Validate Predictions
    for col in confidence_columns:
        df = df[(df[col] >= 0) & (df[col] <= 1)]

    return df

def clean_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Read dataset with {len(df)} rows from {file_path.name}")

        # Standardize text columns
        df = standardize_text_columns(df, ['Tested_Word', 'Original_Word'])

        # Drop rows with null 'Original_Word'
        df = drop_null_values(df, 'Original_Word')

        # Convert specified columns to boolean
        boolean_columns = ['In_Training_Set'] + [f'Top{i}_Is_Valid' for i in range(1, 4)] + [f'Top{i}_Is_Accurate' for i in range(1, 4)]
        df = convert_to_boolean(df, boolean_columns)

        # Convert 'Correct_Letter_Rank' to integer
        df = convert_to_integer(df, 'Correct_Letter_Rank')

        # Remove duplicates
        df = remove_duplicates(df)

        # Validate predicted letters
        predicted_letter_columns = [f'Top{i}_Predicted_Letter' for i in range(1, 4)]
        df = validate_predicted_letters(df, predicted_letter_columns)

        # Additional cleaning steps for confidence scores
        df = clean_dataframe(df)

        # Save the cleaned dataset, overwriting the original file
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
    files_to_clean = [file_path for file_path in data_dir.glob('*.csv') if 'f1' not in file_path.stem]

    for file_path in files_to_clean:
        logger.info(f"Cleaning dataset: {file_path.name}")
        clean_dataset(file_path)

if __name__ == "__main__":
    main()