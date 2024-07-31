import nltk
from nltk.corpus import cmudict, wordnet
from pathlib import Path
import pandas as pd
import re

# Download required resources if not already available
nltk.download('cmudict', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize syllable dictionary from the CMU Pronouncing Dictionary
cmu_dict = cmudict.dict()

# Paths to datasets
DATASET_PATHS = {
    "CLMET3": Path('main/data/outputs/csv/sorted_tokens_clmet_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Lampeter": Path('main/data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Edges": Path('main/data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "CMU": Path('main/data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv'),
    "Brown": Path('main/data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv')
}

def count_syllables(word, cmu_dict):
    """
    Count the number of syllables in a word using a simplified method.
    Handles compound words by splitting them into parts.

    Parameters:
    word (str): The word to count syllables for.
    cmu_dict (dict): The CMU Pronouncing Dictionary.

    Returns:
    int: The number of syllables in the word.
    """
    word = word.lower().strip()
    
    # Handle compound words
    if '-' in word:
        return sum(count_syllables(part, cmu_dict) for part in word.split('-'))
    
    # Check CMU dictionary first
    if word in cmu_dict:
        return len([phoneme for phoneme in cmu_dict[word][0] if phoneme[-1].isdigit()])
    
    # Remove non-alphabetic characters
    word = re.sub(r'[^a-z]', '', word)
    
    # Count vowel groups
    count = len(re.findall(r'[aeiouy]+', word))
    
    # Adjust for common patterns
    if word.endswith('e'):
        count -= 1  # Often silent 'e'
    if word.endswith('le') and len(word) > 2 and word[-3] not in 'aeiou':
        count += 1  # '-le' often forms a syllable
    if word.endswith('es') or word.endswith('ed'):
        count -= 1  # Often these don't add a syllable
    
    # Ensure at least one syllable
    return max(1, count)

def get_part_of_speech(word):
    """
    Get the part of speech for a word using NLTK's POS tagger and map it to WordNet format.

    Parameters:
    word (str): The word to tag.

    Returns:
    str: The part of speech tag in WordNet format.
    """
    # Get the first letter of the POS tag and map it to WordNet format
    tag = nltk.pos_tag([word])[0][1][0].upper()
    # Map the tag to WordNet format
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    # Return the WordNet tag or default to noun
    return tag_dict.get(tag, wordnet.NOUN)

def add_features_to_dataset(data, cmu_dict):
    """
    Add syllable count and part of speech features to a dataset.

    Parameters:
    data (DataFrame): The dataset to add features to.
    cmu_dict (dict): The CMU Pronouncing Dictionary.

    Returns:
    DataFrame: The dataset with added features.
    """
    data['Syllable_Count'] = data['Original_Word'].apply(lambda word: count_syllables(word, cmu_dict))
    data['Part_of_Speech'] = data['Original_Word'].apply(get_part_of_speech)
    return data

def process_datasets(dataset_paths, cmu_dict):
    """
    Process each dataset by reading the CSV file, adding features,
    and saving the updated dataset to a new CSV file.

    Parameters:
    dataset_paths (dict): A dictionary of dataset names and their file paths.
    cmu_dict (dict): The CMU Pronouncing Dictionary.
    """
    for name, path in dataset_paths.items():
        try:
            data = pd.read_csv(path)
            data = add_features_to_dataset(data, cmu_dict)
            output_path = path.parent / f"{name}_with_features.csv"
            data.to_csv(output_path, index=False)
            print(f"Dataset: {name}")
            print(data.head(), "\n")
        except Exception as e:
            print(f"Failed to process dataset {name}: {e}")

# Execute the processing for all datasets
process_datasets(DATASET_PATHS, cmu_dict)