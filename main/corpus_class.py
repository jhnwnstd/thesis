import random
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
    arpa_file = model_directory / f"{corpus_name}_{q}gram.arpa"
    binary_file = model_directory / f"{corpus_name}_{q}gram.klm"

    # Build ARPA model
    if not run_command(['lmplz', '--discount_fallback', '-o', str(q), '--text', str(corpus_path), '--arpa', str(arpa_file)],
                       f"lmplz failed to generate {q}-gram ARPA model for {corpus_name}"):
        return q, None

    # Convert ARPA to binary format
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
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"{error_message}: {e.stderr.decode()}")
        return False

class CorpusManager:
    """
    Manages corpus data, including loading, cleaning, splitting, and model generation.
    """

    CLEAN_PATTERN = reg.compile(r'\b\p{L}+(?:-\p{L}+)*\b') # Regex pattern to extract words
    unique_words_all_corpora = set() # Set of unique words across all corpora

    @staticmethod
    def format_corpus_name(corpus_name) -> str:
        """Format the corpus name by removing redundant parts and file extensions."""
        parts = corpus_name.replace('.txt', '').split('_') # Split corpus name
        return parts[0] if len(parts) > 1 and parts[0] == parts[1] else corpus_name.replace('.txt', '') # Return formatted corpus name

    @staticmethod
    def add_to_global_corpus(unique_words):
        """Add unique words to the global corpus set."""
        CorpusManager.unique_words_all_corpora.update(unique_words) # Update global corpus set

    def __init__(self, corpus_name, config, debug=True):
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
        self.rng = random.Random(config.seed)
        self.corpus = set()
        self.train_set = set()
        self.test_set = set()
        self.all_words = set()
        self.model = {}
        self.load_corpus()
        self.prepare_datasets()
        self.generate_and_load_models()

    def extract_unique_characters(self) -> set:
        """Extract all unique characters from the corpus."""
        return {char for word in self.corpus for char in word} # Extract unique characters

    def clean_text(self, text: str) -> set[str]:
        """
        Clean the input text by extracting words, converting to lowercase,
        and filtering based on minimum word length.
        """
        return {part.lower() for word in self.CLEAN_PATTERN.findall(text)  # Extract words
                for part in word.split('-') # Split hyphenated words
                if len(part) >= self.config.min_word_length} # Filter based on minimum word length

    def load_corpus(self) -> set[str]:
        """
        Load the corpus from a file or NLTK, clean it, and store as a set of unique words.
        """
        file_path = self.config.corpus_dir / f'{self.corpus_name}.txt' # Path to corpus file
        if file_path.is_file(): # Load corpus from file
            with file_path.open('r', encoding='utf-8') as file: # Open file
                self.corpus = self.clean_text(file.read()) # Clean and store corpus
        else:
            try:
                nltk_corpus_name = self.corpus_name.replace('.txt', '') # NLTK corpus name
                nltk.download(nltk_corpus_name, quiet=True) # Download NLTK corpus
                self.corpus = self.clean_text(' '.join(getattr(nltk.corpus, nltk_corpus_name).words())) # Clean and store corpus
            except AttributeError: # NLTK corpus not found
                raise ValueError(f"File '{file_path}' does not exist and NLTK corpus '{nltk_corpus_name}' not found.") # Raise error
            except Exception as e: # Other error
                raise RuntimeError(f"Failed to load corpus '{self.corpus_name}': {e}") # Raise error

        return self.corpus

    def _shuffle_and_split_corpus(self) -> tuple[set[str], set[str]]:
        """
        Shuffle the corpus and split it into training and testing sets.
        """
        shuffled_corpus = list(self.corpus) # Convert to list to shuffle
        self.rng.shuffle(shuffled_corpus) # Shuffle the corpus
        train_size = int(len(self.corpus) * self.config.split_config) # Split the corpus
        return set(shuffled_corpus[:train_size]), set(shuffled_corpus[train_size:]) # Return training and testing sets

    def prepare_datasets(self):
        """
        Prepare training and testing datasets, including letter replacements for the test set.
        """
        self.train_set, unprocessed_test_set = self._shuffle_and_split_corpus() # Split the corpus
        formatted_train_set_path = self.config.sets_dir / f'{self.corpus_name}_formatted_train_set.txt' # Path to formatted training set
        self.generate_formatted_corpus(self.train_set, formatted_train_set_path) # Generate formatted training set

        formatted_test_set = [] # Initialize formatted test set
        for word in unprocessed_test_set: # Iterate over test set
            num_replacements = min(self.config.num_replacements, len(word)) # Number of replacements
            modified_word, missing_letters = self._replace_letters(word, num_replacements) # Replace letters
            if missing_letters: # If letters were replaced
                formatted_test_set.append((modified_word, tuple(missing_letters), word)) # Add to formatted test set

        self.test_set = set(formatted_test_set) # Store formatted test set
        self.all_words = self.train_set.union({original_word for _, _, original_word in self.test_set}) # Store all words

        if self.debug: # Save debug information
            self.save_set_to_file(self.train_set, f'{self.corpus_name}_train_set.txt') # Save training set
            self.save_set_to_file(self.test_set, f'{self.corpus_name}_formatted_test_set.txt') # Save formatted test set
            self.save_set_to_file(self.all_words, f'{self.corpus_name}_all_words.txt') # Save all words

    def generate_formatted_corpus(self, data_set, formatted_corpus_path) -> Path:
        """
        Generate a formatted corpus file from a set of words.

        Args:
            data_set (set): Set of words to format.
            formatted_corpus_path (Path): Path to save the formatted corpus.

        Returns:
            Path: Path to the generated formatted corpus file.
        """
        formatted_text = '\n'.join(' '.join(word) for word in data_set) # Format the text
        with formatted_corpus_path.open('w', encoding='utf-8') as f: # Open file
            f.write(formatted_text) # Write formatted text to file
        return formatted_corpus_path # Return path to formatted corpus

    def generate_models_from_corpus(self, corpus_path):
        """
        Generate KenLM models for different q-gram sizes from the corpus.

        Args:
            corpus_path (Path): Path to the corpus file.
        """
        model_directory = self.config.model_dir / self.corpus_name # Path to model directory
        model_directory.mkdir(parents=True, exist_ok=True) # Create model directory

        model_loaded = False # Initialize model loaded flag
        for q in self.config.q_range: # Iterate over q-gram sizes
            if q not in self.model: # If model does not exist
                _, binary_file = build_kenlm_model(self.corpus_name, q, corpus_path, model_directory) # Build model
                if binary_file: # If model was built
                    self.model[q] = kenlm.Model(binary_file) # Load model
                    model_loaded = True # Set model loaded flag

        if model_loaded: # If model was loaded
            logging.info(f'Model for {q}-gram loaded from {self.corpus_name}') # Log model loaded

    def generate_and_load_models(self):
        """
        Generate formatted corpus and load KenLM models.
        """
        formatted_train_set_path = self.config.sets_dir / f'{self.corpus_name}_formatted_train_set.txt'
        self.generate_formatted_corpus(self.train_set, formatted_train_set_path)
        self.generate_models_from_corpus(formatted_train_set_path)

    def _replace_letters(self, word, num_replacements) -> tuple[str, list[str]]:
        """
        Replace a specified number of letters in a word with underscores.

        Args:
            word (str): The word to modify.
            num_replacements (int): Number of letters to replace.

        Returns:
            tuple: Modified word and list of replaced letters.
        """
        modified_word = word # Initialize modified word
        missing_letters = [] # Initialize list of missing letters
        for _ in range(num_replacements): # Iterate over number of replacements
            if self.has_replaceable_letter(modified_word): # If word has replaceable letter
                modified_word, missing_letter = self._replace_random_letter(modified_word) # Replace random letter
                missing_letters.append(missing_letter) # Add missing letter to list
        return modified_word, missing_letters # Return modified word and list of missing letters

    def _replace_random_letter(self, word) -> tuple[str, str]:
        """
        Replace a random letter in the word with an underscore.

        Args:
            word (str): The word to modify.

        Returns:
            tuple: Modified word and the replaced letter.
        """
        vowel_indices = [i for i, letter in enumerate(word) if Letters.is_vowel(letter)] # Vowel indices
        consonant_indices = [i for i, letter in enumerate(word) if Letters.is_consonant(letter)] # Consonant indices

        if not vowel_indices and not consonant_indices: # If no vowels or consonants
            raise ValueError(f"Unable to replace a letter in word: '{word}'.") # Raise error

        # Choose between vowel and consonant based on replacement ratio
        letter_indices = vowel_indices if self.rng.random() < self.config.vowel_replacement_ratio and vowel_indices else consonant_indices or vowel_indices
        letter_index = self.rng.choice(letter_indices) # Choose random index
        missing_letter = word[letter_index] # Missing letter
        modified_word = word[:letter_index] + '_' + word[letter_index + 1:] # Replace letter with underscore

        return modified_word, missing_letter
    
    def has_replaceable_letter(self, word) -> bool:
        """
        Check if the word has any replaceable letters (vowels or consonants).

        Args:
            word (str): The word to check.

        Returns:
            bool: True if the word has replaceable letters, False otherwise.
        """
        return any(Letters.is_vowel(letter) for letter in word) or any(Letters.is_consonant(letter) for letter in word)

    def save_set_to_file(self, data_set, file_name):
        """
        Save a set of data to a file.

        Args:
            data_set (set): Set of data to save.
            file_name (str): Name of the file to save the data to.
        """
        file_path = self.config.sets_dir / file_name
        with file_path.open('w', encoding='utf-8') as file:
            file.writelines(f"{item}\n" for item in data_set)