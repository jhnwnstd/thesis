import pandas as pd
from pathlib import Path
import logging
from scipy.stats import zscore

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def rename_columns(df, columns_map):
    """Rename columns based on the provided columns_map dictionary."""
    df.rename(columns=columns_map, inplace=True)
    return df

def standardize_text_columns(df, columns):
    """Standardize text columns by converting to lowercase and stripping whitespace."""
    df[columns] = df[columns].apply(lambda x: x.str.lower().str.strip())
    return df

def drop_null_values(df, column):
    """Drop rows with null values in the specified column and log the number of dropped rows."""
    initial_count = len(df)
    df.dropna(subset=[column], inplace=True)
    dropped_rows = initial_count - len(df)
    if dropped_rows > 0:
        logger.info(f"Dropped {dropped_rows} rows with null '{column}'. New row count: {len(df)}")
    return df

def convert_columns(df, columns, dtype):
    """Convert specified columns to the given data type."""
    df[columns] = df[columns].astype(dtype)
    return df

def remove_duplicates(df):
    """Remove duplicate rows and log the number of removed duplicates."""
    initial_row_count = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = initial_row_count - len(df)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate rows. New row count: {len(df)}")
    return df

def validate_predicted_letters(df, columns):
    """Validate that the specified columns contain exactly one letter."""
    for col in columns:
        df = df[df[col].apply(lambda x: isinstance(x, str) and len(x) == 1)]
    return df

def remove_outliers_conservatively(df, columns, z_threshold=3):
    """Remove rows with z-scores beyond the specified threshold for any of the specified columns."""
    z_scores = df[columns].apply(zscore)
    df_cleaned = df[(z_scores < z_threshold).all(axis=1)]
    return df_cleaned

def clean_dataframe(df):
    """Perform comprehensive data cleaning on the DataFrame."""
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
    """Read, clean, and overwrite the dataset at the specified file path."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Read {len(df)} rows from {file_path.name}")

        # Rename columns as needed
        df = rename_columns(df, {'# Tested_Word': 'Tested_Word'})

        # Perform standard cleaning operations
        df = standardize_text_columns(df, ['Tested_Word', 'Original_Word'])
        df = drop_null_values(df, 'Original_Word')
        boolean_columns = ['In_Training_Set'] + [f'Top{i}_Is_Valid' for i in range(1, 4)] + [f'Top{i}_Is_Accurate' for i in range(1, 4)]
        df = convert_columns(df, boolean_columns, bool)
        df = convert_columns(df, ['Correct_Letter_Rank'], int)
        df = validate_predicted_letters(df, [f'Top{i}_Predicted_Letter' for i in range(1, 4)])
        df = clean_dataframe(df)
        
        df.to_csv(file_path, index=False)
        
    except pd.errors.EmptyDataError:
        logger.error(f"Error: {file_path.name} is empty.")
    except pd.errors.ParserError:
        logger.error(f"Error: Could not parse {file_path.name}.")
    except KeyError as e:
        logger.error(f"Error: Missing expected column in {file_path.name}: {e}")
    except ValueError as e:
        logger.error(f"Error: Data type conversion issue in {file_path.name}: {e}")
    except Exception as e:
        logger.error(f"Error: Failed to clean {file_path.name} due to {e}")

def main():
    """Main function to clean all CSV files in the specified directory."""
    data_dir = Path('main/data/outputs/csv')
    files_to_clean = list(data_dir.glob('*.csv'))

    for file_path in files_to_clean:
        logger.info(f"Processing file: {file_path.name}")
        clean_dataset(file_path)

if __name__ == "__main__":
    main()
