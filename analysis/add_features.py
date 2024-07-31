import spacy
import nltk
from nltk.corpus import cmudict, wordnet
from pathlib import Path
import pandas as pd
import re
import logging
import warnings

# Configure logging to display messages
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Download required NLTK resources if not already available
resources = ['cmudict', 'averaged_perceptron_tagger', 'wordnet', 'punkt', 'stopwords']
for resource in resources:
    nltk.download(resource, quiet=True)

# Initialize spaCy transformer model for NLP tasks
nlp = spacy.load("en_core_web_sm")

# Initialize CMU Pronouncing Dictionary for syllable counting
cmu_dict = cmudict.dict()

# Precompile regex patterns
vowel_pattern = re.compile(r'[aeiouy]+')
non_alpha_pattern = re.compile(r'[^a-z]')

# Suppress specific openpyxl warnings related to unsupported extensions
warnings.filterwarnings("ignore", category=UserWarning, module='openpyxl')

def load_morpheme_data(path):
    """
    Load morpheme data from an Excel file into a dictionary.

    Parameters:
    path (str or Path): Path to the Excel file.

    Returns:
    dict: Dictionary with words as keys and their morpheme counts as values.
    """
    try:
        # Read all sheets from the Excel file into a dictionary of DataFrames
        morpheme_data = pd.read_excel(path, sheet_name=None)
        
        # Combine all DataFrames into a single DataFrame
        all_morphemes = pd.concat(morpheme_data.values(), ignore_index=True)
        
        # Create a dictionary mapping words to their morpheme counts
        return dict(zip(all_morphemes['Word'].str.lower(), all_morphemes['Nmorph']))
    except Exception as e:
        logging.error(f"Failed to load morpheme data from {path}: {e}")
        return {}

def count_syllables(word, cmu_dict):
    """
    Count the number of syllables in a word.

    Parameters:
    word (str): The word to count syllables for.
    cmu_dict (dict): The CMU Pronouncing Dictionary.

    Returns:
    int: The number of syllables in the word.
    """
    word = word.lower().strip()
    
    # Handle compound words by splitting and counting syllables in each part
    if '-' in word:
        return sum(count_syllables(part, cmu_dict) for part in word.split('-'))
    
    # Use the CMU dictionary to count syllables
    if word in cmu_dict:
        return len([phoneme for phoneme in cmu_dict[word][0] if phoneme[-1].isdigit()])
    
    # Remove non-alphabetic characters
    word = non_alpha_pattern.sub('', word)
    
    # Count vowel groups as syllables
    count = len(vowel_pattern.findall(word))
    
    # Adjust for common patterns
    if word.endswith('e'):
        count -= 1 # Ignore silent 'e'
    if word.endswith('le') and len(word) > 2 and word[-3] not in 'aeiouy':
        count += 1 # Count 'le' as a syllable
    
    # Ensure at least one syllable
    return max(1, count)

def get_part_of_speech(word):
    """
    Get the part of speech for a word using spaCy's transformer model and map it to WordNet format.

    Parameters:
    word (str): The word to tag.

    Returns:
    str: The part of speech tag in WordNet format.
    """
    # Use spaCy to get the part of speech
    doc = nlp(word)
    spacy_pos = doc[0].pos_
    
    # Map spaCy POS tags to WordNet POS tags
    tag_dict = {
        "ADJ": wordnet.ADJ,
        "NOUN": wordnet.NOUN,
        "VERB": wordnet.VERB,
        "ADV": wordnet.ADV
    }
    
    # Return the corresponding WordNet POS tag or default to NOUN
    return tag_dict.get(spacy_pos, wordnet.NOUN)

def count_morphemes(word, morpheme_dict):
    """
    Count derivational morphemes in a word using the provided morpheme data.

    Parameters:
    word (str): The word to count morphemes for.
    morpheme_dict (dict): Dictionary with morpheme counts for words.

    Returns:
    int: The number of morphemes in the word.
    """
    # Return the morpheme count from the dictionary, defaulting to 1 if not found
    return morpheme_dict.get(word.lower(), 1)

def add_features_to_dataset(data, cmu_dict, morpheme_dict):
    """
    Add syllable count, part of speech, and morpheme count features to a dataset.

    Parameters:
    data (DataFrame): The dataset to add features to.
    cmu_dict (dict): The CMU Pronouncing Dictionary.
    morpheme_dict (dict): Dictionary with morpheme counts for words.

    Returns:
    DataFrame: The dataset with added features.
    """
    # Apply functions to add new features to the DataFrame
    data['Syllable_Count'] = data['Original_Word'].apply(lambda word: count_syllables(word, cmu_dict))
    data['Part_of_Speech'] = data['Original_Word'].apply(get_part_of_speech)
    data['Morpheme_Count'] = data['Original_Word'].apply(lambda word: count_morphemes(word, morpheme_dict))
    
    return data

def process_datasets(dataset_paths, cmu_dict, morpheme_dict):
    """
    Process each dataset by reading the CSV file, adding features,
    and saving the updated dataset to a new CSV file.

    Parameters:
    dataset_paths (dict): A dictionary of dataset names and their file paths.
    cmu_dict (dict): The CMU Pronouncing Dictionary.
    morpheme_dict (dict): Dictionary with morpheme counts for words.
    """
    for name, path in dataset_paths.items():
        try:
            # Read the dataset from CSV
            data = pd.read_csv(path)
            
            # Add features to the dataset
            data = add_features_to_dataset(data, cmu_dict, morpheme_dict)
            
            # Save the updated dataset to a new CSV file
            output_path = path.parent / f"{name}_with_features.csv"
            data.to_csv(output_path, index=False)
            
            # Log the processing result
            logging.info(f"Processed dataset: {name}")
            logging.info(data.head())
        except Exception as e:
            # Log any errors that occur during processing
            logging.warning(f"Failed to process dataset {name}: {e}")

if __name__ == "__main__":
    # Define the paths to the datasets
    DATASET_PATHS = {
        "CLMET3": Path('main/data/outputs/csv/sorted_tokens_clmet_context_sensitive_split0.5_qrange7-7_prediction.csv'),
        "Lampeter": Path('main/data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv'),
        "Edges": Path('main/data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv'),
        "CMU": Path('main/data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv'),
        "Brown": Path('main/data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv')
    }

    # Define the path to the morpheme data file
    morpheme_data_path = Path('main/data/MorphoLEX_en.xlsx')
    
    # Load morpheme data
    morpheme_dict = load_morpheme_data(morpheme_data_path)
    
    # Process the datasets
    process_datasets(DATASET_PATHS, cmu_dict, morpheme_dict)