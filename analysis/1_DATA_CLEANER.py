import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def clean_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Read dataset with {len(df)} rows from {file_path.name}")
        
        # Standardize 'Tested_Word' and 'Original_Word'
        df['Tested_Word'] = df['Tested_Word'].str.lower().str.strip()
        df['Original_Word'] = df['Original_Word'].str.lower().str.strip()
        
        # Drop rows where 'Original_Word' is null
        df.dropna(subset=['Original_Word'], inplace=True)
        logger.info(f"Dropped rows with null 'Original_Word'. New row count: {len(df)}")
        
        # Ensure boolean data types for 'In_Training_Set' and accuracy/validation flags
        boolean_columns = ['In_Training_Set'] + [f'Top{i}_Is_Valid' for i in range(1, 4)] + \
                          [f'Top{i}_Is_Accurate' for i in range(1, 4)]
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)

        # Convert 'Correct_Letter_Rank' to integer
        if 'Correct_Letter_Rank' in df.columns:
            df['Correct_Letter_Rank'] = df['Correct_Letter_Rank'].astype(int)

        # Remove duplicates
        initial_row_count = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_row_count - len(df)
        logger.info(f"Removed {duplicates_removed} duplicate rows. New row count: {len(df)}")
        
        # Validation checks (e.g., predicted letters are single characters)
        for i in range(1, 4):
            predicted_letter_col = f'Top{i}_Predicted_Letter'
            if predicted_letter_col in df.columns:
                df = df[df[predicted_letter_col].apply(lambda x: isinstance(x, str) and len(x) == 1)]
        
        # Standardize 'Top1_Confidence' column to be in range [0, 1]
        if 'Top1_Confidence' in df.columns:
            df['Top1_Confidence'] = df['Top1_Confidence'].apply(lambda x: min(max(x, 0), 1))

        # Save the cleaned dataset, overwriting the original file
        df.to_csv(file_path, index=False)
        
        logger.info(f"Cleaned dataset overwritten at {file_path}. Duplicates removed: {duplicates_removed}.")
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

data_dir = Path('main/data/outputs/csv')

files_to_clean = [file_path for file_path in data_dir.glob('*.csv') if 'f1' not in file_path.stem]

for file_path in files_to_clean:
    logger.info(f"Cleaning dataset: {file_path.name}")
    clean_dataset(file_path)