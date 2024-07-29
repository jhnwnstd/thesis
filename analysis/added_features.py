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

# Initialize syllable dictionary
cmu_dict = cmudict.dict()

# Paths to datasets
DATASET_PATHS = {
    "CLMET3": 'main/data/outputs/csv/sorted_tokens_clmet_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Lampeter": 'main/data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Edges": 'main/data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "CMU": 'main/data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Brown": 'main/data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv'
}

# Function to count syllables using fallback mechanism
def count_syllables(word, cmu_dict):
    word = word.lower()
    if word in cmu_dict:
        return len([y for y in cmu_dict[word][0] if y[-1].isdigit()])
    else:
        word = re.sub(r'[^a-zA-Z]', '', word)  # Remove non-alphabetic characters
        return len(re.findall(r'[aeiouy]+', word.lower()))  # Count vowel groups as syllables

# Function to get part of speech
def get_part_of_speech(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Function to add features to dataset
def add_features_to_dataset(data, cmu_dict):
    data['Syllable_Count'] = data['Original_Word'].apply(lambda word: count_syllables(word, cmu_dict))
    data['Part_of_Speech'] = data['Original_Word'].apply(get_part_of_speech)
    return data

# Function to process and save datasets
def process_datasets(dataset_paths, cmu_dict):
    for name, path in dataset_paths.items():
        data = pd.read_csv(path)
        data = add_features_to_dataset(data, cmu_dict)
        output_path = f'main/data/outputs/csv/{name}_with_features.csv'
        data.to_csv(output_path, index=False)
        print(f"Dataset: {name}")
        print(data.head(), "\n")

# Execute the processing
process_datasets(DATASET_PATHS, cmu_dict)