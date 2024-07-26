import numpy as np
import logging
import regex as reg
from pathlib import Path
import subprocess
from enum import Enum
import nltk
import kenlm

class Letters(Enum):
    """
    Enum class defining sets of vowels and consonants.
    Provides methods to check if a character is a vowel or consonant.
    """
    VOWELS = 'aeèéiîouyæœ'
    CONSONANTS = 'bcdfghjklmnpqrstvwxzȝ'

    @staticmethod
    def is_vowel(char):
        """Check if a character is a vowel."""
        return char in Letters.VOWELS.value

    @staticmethod
    def is_consonant(char):
        """Check if a character is a consonant."""
        return char in Letters.CONSONANTS.value

def build_kenlm_model(corpus_name, q, corpus_path, model_directory) -> tuple[int, str]:
    """
    Build a KenLM language model for a specified q-gram size.

    Args:
        corpus_name (str): Name of the corpus.
        q (int): Size of the q-gram.
        corpus_path (Path): Path to the corpus file.
        model_directory (Path): Directory to store the model.

    Returns:
        tuple: q-gram size and path to the binary model file, or (q, None) if build fails.
    """
    # Define paths for intermediate ARPA file and final binary KenLM model
    arpa_file = model_directory / f"{corpus_name}_{q}gram.arpa"
    binary_file = model_directory / f"{corpus_name}_{q}gram.klm"

    # Build ARPA model using KenLM's lmplz tool
    # The '--discount_fallback' option ensures smoother probability estimates
    if not run_command(['lmplz', '--discount_fallback', '-o', str(q), '--text', str(corpus_path), '--arpa', str(arpa_file)],
                       f"lmplz failed to generate {q}-gram ARPA model for {corpus_name}"):
        return q, None

    # Convert ARPA to binary format for faster loading and reduced memory usage
    if not run_command(['build_binary', '-s', str(arpa_file), str(binary_file)],
                       f"build_binary failed to convert {q}-gram ARPA model to binary format for {corpus_name}"):
        return q, None

    return q, str(binary_file)

def run_command(command, error_message):
    """
    Execute a shell command and handle potential errors.

    Args:
        command (list): Command to execute.
        error_message (str): Error message to log if command fails.

    Returns:
        bool: True if command succeeds, False otherwise.
    """
    try:
        # Run the command, suppressing stdout but capturing stderr
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError as e:
        # Log the error message along with the captured stderr output
        logging.error(f"{error_message}: {e.stderr.decode()}")
        return False

class CorpusManager:
    """
    Manages corpus data, including loading, cleaning, splitting, and model generation.
    """

    # Regex pattern to extract words, including hyphenated words
    CLEAN_PATTERN = reg.compile(r'\b\p{L}+(?:-\p{L}+)*\b')
    
    # Set to store unique words across all corpora, useful for cross-corpus analysis
    unique_words_all_corpora = set()

    @staticmethod
    def format_corpus_name(corpus_name) -> str:
        """Format the corpus name by removing redundant parts and file extensions."""
        # Split the corpus name and remove .txt extension
        parts = corpus_name.replace('.txt', '').split('_')
        # If the first two parts are identical, use only the first part; otherwise, use the full name without extension
        return parts[0] if len(parts) > 1 and parts[0] == parts[1] else corpus_name.replace('.txt', '')

    @staticmethod
    def add_to_global_corpus(unique_words):
        """Add unique words to the global corpus set for cross-corpus analysis."""
        CorpusManager.unique_words_all_corpora.update(unique_words)

    def __init__(self, corpus_name, config, debug=False):
        """
        Initialize the CorpusManager with a specific corpus and configuration.

        Args:
            corpus_name (str): Name of the corpus.
            config: Configuration object containing various settings.
            debug (bool): Whether to save debug information.
        """
        self.corpus_name = self.format_corpus_name(corpus_name)
        self.config = config
        self.debug = debug
        # Use NumPy's random number generator for better randomization control
        self.rng = np.random.RandomState(config.seed)
        # Initialize corpus, training set, test set, and all words as NumPy arrays
        self.corpus = np.array([])
        self.train_set = np.array([])
        self.test_set = np.array([])
        self.all_words = np.array([])
        self.model = {}
        self.load_corpus()  # Load and clean the corpus
        self.prepare_datasets()  # Prepare training and testing datasets
        self.generate_and_load_models()  # Generate and load KenLM models

    def extract_unique_characters(self) -> set:
        """Extract all unique characters from the corpus for character-level analysis."""
        return {char for word in self.corpus for char in word}

    def clean_text(self, text: str) -> set[str]:
        """
        Clean the input text by extracting words, converting to lowercase,
        and filtering based on minimum word length.
        """
        words = self.CLEAN_PATTERN.findall(text.lower())  # Find all words and convert to lowercase
        split_words = {word for part in words for word in part.split('-') if len(word) >= self.config.min_word_length}
        return split_words

    def load_corpus(self) -> np.ndarray:
        """
        Load the corpus from a file or NLTK, clean it, and store as a sorted NumPy array of unique words.
        """
        file_path = self.config.corpus_dir / f'{self.corpus_name}.txt'  # Path to the local corpus file
        if file_path.is_file():
            # Load corpus from local file if it exists
            with file_path.open('r', encoding='utf-8') as file:
                corpus = self.clean_text(file.read())
        else:
            # Attempt to load corpus from NLTK if local file doesn't exist
            try:
                nltk_corpus_name = self.corpus_name.replace('.txt', '')
                nltk.download(nltk_corpus_name, quiet=True)  # Download NLTK corpus if not already present
                corpus = self.clean_text(' '.join(getattr(nltk.corpus, nltk_corpus_name).words()))
            except AttributeError:
                # Raise error if neither local file nor NLTK corpus is found
                raise ValueError(f"File '{file_path}' does not exist and NLTK corpus '{nltk_corpus_name}' not found.")
            except Exception as e:
                # Catch and re-raise any other exceptions during corpus loading
                raise RuntimeError(f"Failed to load corpus '{self.corpus_name}': {e}")

        # Store the cleaned corpus as a sorted NumPy array of unique words
        self.corpus = np.array(sorted(corpus))
        return self.corpus

    def _shuffle_and_split_corpus(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Deterministically shuffle the corpus and split it into training and testing sets.
        """
        # Generate a permutation of indices using NumPy's random number generator
        shuffled_indices = self.rng.permutation(len(self.corpus))
        # Reorder the corpus based on shuffled indices
        shuffled_corpus = self.corpus[shuffled_indices]
        # Calculate the split point
        split_point = int(len(shuffled_corpus) * self.config.split_config)
        # Split into training and testing sets
        return shuffled_corpus[:split_point], shuffled_corpus[split_point:]

    def prepare_datasets(self):
        """
        Prepare training and testing datasets, including letter replacements for the test set.
        """
        # Shuffle and split the corpus
        self.train_set, unprocessed_test_set = self._shuffle_and_split_corpus()
        
        # Define the path for the formatted training set file
        formatted_train_set_path = self.config.sets_dir / f'{self.corpus_name}_formatted_train_set.txt'
        # Generate the formatted corpus file for the training set
        self.generate_formatted_corpus(self.train_set, formatted_train_set_path)

        # Use NumPy's vectorize to apply _replace_letters to each word in the test set
        vectorized_replace = np.vectorize(self._replace_letters, otypes=[object, object])
        modified_words, missing_letters = vectorized_replace(unprocessed_test_set, self.config.num_replacements)
        
        # Store the formatted test set as a NumPy array of tuples
        self.test_set = np.empty(len(unprocessed_test_set), dtype=object)
        self.test_set[:] = list(zip(modified_words, missing_letters, unprocessed_test_set))
        
        # Combine the training set and the original words from the test set to create the complete set of all words
        self.all_words = np.unique(np.concatenate([self.train_set, unprocessed_test_set]))

        if self.debug:
            # Save the training set, test set, and all words set to files if debugging is enabled
            self.save_set_to_file(self.train_set, f'{self.corpus_name}_train_set.txt')
            self.save_set_to_file(self.test_set, f'{self.corpus_name}_formatted_test_set.txt')
            self.save_set_to_file(self.all_words, f'{self.corpus_name}_all_words.txt')

    def _replace_letters(self, word, num_replacements):
        """
        Replace a specified number of letters in a word with underscores.

        Args:
            word (str): The word to modify.
            num_replacements (int): Number of letters to replace.

        Returns:
            tuple: Modified word and list of replaced letters.
        """
        char_array = np.array(list(word))  # Convert word to a NumPy array of characters
        replaceable_mask = np.isin(char_array, list(Letters.VOWELS.value + Letters.CONSONANTS.value))  # Create mask
        replaceable_indices = np.where(replaceable_mask)[0]  # Get indices of replaceable letters

        if len(replaceable_indices) == 0:
            return word, []

        num_to_replace = min(num_replacements, len(replaceable_indices))
        replace_indices = self.rng.choice(replaceable_indices, size=num_to_replace, replace=False)

        missing_letters = char_array[replace_indices].tolist()  # List of letters that will be replaced
        char_array[replace_indices] = '_'  # Replace letters with underscores

        return ''.join(char_array), missing_letters

    def generate_formatted_corpus(self, data_set, formatted_corpus_path) -> Path:
        """
        Generate a formatted corpus file from a set of words.

        Args:
            data_set (np.ndarray): Set of words to format.
            formatted_corpus_path (Path): Path to save the formatted corpus.

        Returns:
            Path: Path to the generated formatted corpus file.
        """
        # Join words with spaces and separate by newlines for KenLM compatibility
        formatted_text = '\n'.join(' '.join(word) for word in data_set)
        formatted_corpus_path.write_text(formatted_text, encoding='utf-8')
        return formatted_corpus_path

    def generate_models_from_corpus(self, corpus_path):
        """
        Generate KenLM models for different q-gram sizes from the corpus.

        Args:
            corpus_path (Path): Path to the corpus file.
        """
        model_directory = self.config.model_dir / self.corpus_name  # Directory to store the models
        model_directory.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

        for q in self.config.q_range:
            if q not in self.model:
                # Build and load KenLM model for each q-gram size
                _, binary_file = build_kenlm_model(self.corpus_name, q, corpus_path, model_directory)
                if binary_file:
                    self.model[q] = kenlm.Model(binary_file)  # Store the loaded model in the dictionary

        if self.model:
            logging.info(f'Models loaded for {self.corpus_name}')  # Log a message if models are loaded

    def generate_and_load_models(self):
        """
        Generate formatted corpus and load KenLM models.
        """
        formatted_train_set_path = self.config.sets_dir / f'{self.corpus_name}_formatted_train_set.txt'  # Path for formatted training set
        self.generate_formatted_corpus(self.train_set, formatted_train_set_path)  # Generate formatted corpus file
        self.generate_models_from_corpus(formatted_train_set_path)  # Generate and load models from the formatted corpus

    def save_set_to_file(self, data_set, file_name):
        """
        Save a set of data to a file.

        Args:
            data_set (np.ndarray): Set of data to save.
            file_name (str): Name of the file to save the data to.
        """
        file_path = self.config.sets_dir / file_name  # Path to the file

        if np.issubdtype(data_set.dtype, np.number):
            np.savetxt(file_path, data_set, delimiter=',')
        else:
            with file_path.open('w', encoding='utf-8') as file:
                if isinstance(data_set[0], tuple):
                    lines = ['{},{}'.format(*map(str, item)) if len(item) == 2 else ','.join(map(str, item)) for item in data_set]
                else:
                    lines = list(map(str, data_set))
                file.write('\n'.join(lines) + '\n')
