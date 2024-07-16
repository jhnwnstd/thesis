import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def clean_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Read dataset with {len(df)} rows from {file_path.name}")
        
        # Standardize 'Tested_Word' and 'Original_Word'
        df['Tested_Word'] = df['Tested_Word'].str.lower().str.strip()
        df['Original_Word'] = df['Original_Word'].str.lower().str.strip()
        
        # Drop rows where 'Original_Word' is null
        df.dropna(subset=['Original_Word'], inplace=True)
        logging.info(f"Dropped rows with null 'Original_Word'. New row count: {len(df)}")
        
        # Ensure boolean data types for 'In_Training_Set' and accuracy/validation flags
        boolean_columns = ['In_Training_Set'] + [f'Top{i}_Is_Valid' for i in range(1, 4)] + \
                          [f'Top{i}_Is_Accurate' for i in range(1, 4)]
        df[boolean_columns] = df[boolean_columns].astype(bool)

        # Convert 'Correct_Letter_Rank' to integer
        df['Correct_Letter_Rank'] = df['Correct_Letter_Rank'].astype(int)

        # Remove duplicates
        initial_row_count = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_row_count - len(df)
        logging.info(f"Removed {duplicates_removed} duplicate rows. New row count: {len(df)}")
        
        # Validation checks (e.g., predicted letters are single characters)
        for i in range(1, 4):
            df = df[df[f'Top{i}_Predicted_Letter'].apply(lambda x: len(str(x)) == 1)]
        
        # Filter by confidence threshold (example threshold: 0.01)
        confidence_threshold = 0.1
        df = df[df['Top1_Confidence'] >= confidence_threshold]
        logging.info(f"Filtered rows with 'Top1_Confidence' below {confidence_threshold}. New row count: {len(df)}")
        
        # Save the cleaned dataset, overwriting the original file
        df.to_csv(file_path, index=False)
        
        logging.info(f"Cleaned dataset overwritten at {file_path}. Duplicates removed: {duplicates_removed}.")
    except pd.errors.EmptyDataError:
        logging.error(f"{file_path.name} is empty.")
    except pd.errors.ParserError:
        logging.error(f"{file_path.name} could not be parsed.")
    except KeyError as e:
        logging.error(f"Missing expected column in {file_path.name}: {e}")
    except ValueError as e:
        logging.error(f"Data type conversion error in {file_path.name}: {e}")
    except Exception as e:
        logging.error(f"Failed to clean {file_path.name} due to {e}")

data_dir = Path('main/data/outputs/csv')

files_to_clean = [file_path for file_path in data_dir.glob('*.csv') if 'f1' not in file_path.stem]

for file_path in files_to_clean:
    logging.info(f"Cleaning dataset: {file_path.name}")
    clean_dataset(file_path)
