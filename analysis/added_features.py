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
    "CLMET3": 'main/data/outputs/csv/sorted_tokens_clmet_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Lampeter": 'main/data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Edges": 'main/data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "CMU": 'main/data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Brown": 'main/data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv'
}

def count_syllables(word, cmu_dict):
    """
    Count the number of syllables in a word using the CMU Pronouncing Dictionary
    as the primary method and a fallback method based on vowel group counting.

    Parameters:
    word (str): The word to count syllables for.
    cmu_dict (dict): The CMU Pronouncing Dictionary.

    Returns:
    int: The number of syllables in the word.
    """
    word = word.lower()
    if word in cmu_dict:
        # Use the CMU dictionary to count syllables
        return len([y for y in cmu_dict[word][0] if y[-1].isdigit()])
    else:
        # Remove non-alphabetic characters and count vowel groups as syllables
        word = re.sub(r'[^a-zA-Z]', '', word)
        return len(re.findall(r'[aeiouy]+', word.lower()))

def get_part_of_speech(word):
    """
    Get the part of speech for a word using NLTK's POS tagger and map it to WordNet format.

    Parameters:
    word (str): The word to tag.

    Returns:
    str: The part of speech tag in WordNet format.
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
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
    # Add syllable count feature
    data['Syllable_Count'] = data['Original_Word'].apply(lambda word: count_syllables(word, cmu_dict))
    # Add part of speech feature
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
        # Read the dataset from CSV
        data = pd.read_csv(path)
        # Add features to the dataset
        data = add_features_to_dataset(data, cmu_dict)
        # Define the output path for the updated dataset
        output_path = f'main/data/outputs/csv/{name}_with_features.csv'
        # Save the updated dataset to a new CSV file
        data.to_csv(output_path, index=False)
        # Print a summary of the dataset
        print(f"Dataset: {name}")
        print(data.head(), "\n")

# Execute the processing for all datasets
process_datasets(DATASET_PATHS, cmu_dict)