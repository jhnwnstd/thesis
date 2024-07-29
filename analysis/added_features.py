import nltk
from nltk.corpus import cmudict
from nltk.corpus import wordnet
from pathlib import Path
import pandas as pd
import re

# Download required resources manually once
nltk.download('cmudict')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

# Initialize syllable dictionary
d = cmudict.dict()

# Paths to datasets
DATASET_PATHS = {
    "CLMET3": 'main/data/outputs/csv/sorted_tokens_clmet_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Lampeter": 'main/data/outputs/csv/sorted_tokens_lampeter_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Edges": 'main/data/outputs/csv/sorted_tokens_openEdges_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "CMU": 'main/data/outputs/csv/cmudict_context_sensitive_split0.5_qrange7-7_prediction.csv',
    "Brown": 'main/data/outputs/csv/brown_context_sensitive_split0.5_qrange7-7_prediction.csv'
}

# Function to count syllables using fallback mechanism
def count_syllables(word):
    try:
        # Try to get the syllable count from CMU dictionary
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    except KeyError:
        # Fallback: Count vowel groups as an estimate of syllable count
        word = re.sub(r'[^a-zA-Z]', '', word)  # Remove non-alphabetic characters
        return len(re.findall(r'[aeiouy]+', word.lower())) # Count vowel groups as syllables

# Function to get part of speech
def get_part_of_speech(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Function to add features to dataset
def add_features_to_dataset(data):
    data['Syllable_Count'] = data['Original_Word'].apply(count_syllables)
    data['Part_of_Speech'] = data['Original_Word'].apply(get_part_of_speech)
    return data

# Process all datasets
for name, path in DATASET_PATHS.items():
    # Load dataset
    data = pd.read_csv(path)
    
    # Add features to the dataset
    data = add_features_to_dataset(data)
    
    # Save the updated dataset
    output_path = f'main/data/outputs/csv/{name}_with_features.csv'
    data.to_csv(output_path, index=False)
    
    # Display the first few rows of the updated dataset
    print(f"Dataset: {name}")
    print(data.head(), "\n")