import logging
import random
import regex as reg
import subprocess
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import List, Callable, Tuple

import kenlm
import nltk
from evaluation_class import EvaluateModel

class Letters(Enum):
    """
    Enum class defining sets of vowels and consonants.
    Provides methods to check if a character is a vowel or consonant.
    """
    VOWELS = 'aeèéiîouyæœ'
    CONSONANTS = 'bcdfghjklmnpqrstvwxzȝ'

    @staticmethod
    def is_vowel(char: str) -> bool:
        """Check if a character is a vowel."""
        return char in Letters.VOWELS.value

    @staticmethod
    def is_consonant(char: str) -> bool:
        """Check if a character is a consonant."""
        return char in Letters.CONSONANTS.value

def build_kenlm_model(corpus_name: str, q: int, corpus_path: Path, model_directory: Path) -> Tuple[int, str]:
    """
    Builds KenLM language models for specified q-gram sizes.
    Generates an ARPA file and then converts it to a binary format for efficiency.

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

    # Attempt to build the ARPA model file
    if not run_command(['lmplz', '--discount_fallback', '-o', str(q), 
                        '--text', str(corpus_path), '--arpa', str(arpa_file)],
                       "lmplz failed to generate ARPA model"):
        return q, None

    # Attempt to convert the ARPA model to binary format
    if not run_command(['build_binary', '-s', str(arpa_file), str(binary_file)],
                       "build_binary failed to convert ARPA model to binary format"):
        return q, None

    return q, str(binary_file)

def run_command(command: List[str], error_message: str) -> bool:
    """
    Executes a command as a subprocess and logs any errors encountered.

    Args:
        command (list): Command to execute.
        error_message (str): Error message to log if command fails.

    Returns:
        bool: True if the command executes successfully, False if an error occurs.
    """
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"{error_message}: {e.stderr.decode()}")
        return False

class Config:
    """Configuration class for language model testing parameters."""

    def __init__(self, base_dir: str = None):
        """
        Initialize the Config object with default values and directory structures.

        Args:
            base_dir (str, optional): Base directory for the project. Defaults to the parent directory of this file.
        """
        self.base_dir = Path(base_dir if base_dir else __file__).parent
        self._setup_directories()
        self._set_default_values()

    def _setup_directories(self):
        """Set up the directory structure for the project."""
        self.data_dir = self.base_dir / 'data'
        self.model_dir = self.data_dir / 'models'
        self.log_dir = self.data_dir / 'logs'
        self.corpus_dir = self.data_dir / 'corpora'
        self.output_dir = self.data_dir / 'outputs'
        self.text_dir = self.output_dir / 'texts'
        self.csv_dir = self.output_dir / 'csv'
        self.sets_dir = self.output_dir / 'sets'
        self.directories = [
            self.data_dir, self.model_dir, self.log_dir, self.corpus_dir,
            self.output_dir, self.sets_dir, self.text_dir, self.csv_dir
        ]

    def _set_default_values(self):
        """Set default values for configuration parameters."""
        self.seed = 42
        self.q_range = [7, 7]
        self.split_config = 0.5
        self.vowel_replacement_ratio = 0.2
        self.consonant_replacement_ratio = 0.8
        self.min_word_length = 3
        self.num_replacements = 1
        self.prediction_method_name = 'context_sensitive'
        self.log_level = logging.INFO

    def setup_logging(self):
        """Set up logging with file and console handlers."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logfile = self.log_dir / 'logfile.log'
        logging.basicConfig(
            level=self.log_level,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(logfile, mode='a'),
                logging.StreamHandler()
            ]
        )

    def create_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in self.directories:
            directory.mkdir(parents=True, exist_ok=True)

class CorpusManager:
    """Manages corpus data, including loading, cleaning, splitting, and model generation."""

    CLEAN_PATTERN = reg.compile(r'\b\p{L}+(?:-\p{L}+)*\b')
    corpora_tokens = []  # List to store all words across corpora

    @staticmethod
    def add_to_global_corpus(words: List[str]):
        """Add words to the global corpus list."""
        CorpusManager.corpora_tokens.extend(words)

    @staticmethod
    def format_corpus_name(corpus_name: str) -> str:
        """Format the corpus name by removing redundant parts and file extensions."""
        parts = corpus_name.replace('.txt', '').split('_')
        return parts[0] if len(parts) > 1 and parts[0] == parts[1] else corpus_name.replace('.txt', '')

    def __init__(self, corpus_name: str, config: Config, split_type: str = 'A', debug: bool = True):
        """
        Initialize the CorpusManager with a specific corpus and configuration.

        Args:
            corpus_name (str): Name of the corpus.
            config (Config): Configuration object.
            split_type (str): Type of dataset split ('A', 'B', or 'HAPAX').
            debug (bool): Whether to save debug information.
        """
        self.corpus_name = self.format_corpus_name(corpus_name)
        self.config = config
        self.split_type = split_type
        self.debug = debug
        self.rng = random.Random(config.seed)
        self.corpus = Counter()
        self.load_corpus()
        self.train_set = []
        self.test_set = []
        self.model = {}
        self.all_words = set()
        self.prepare_datasets()

    def extract_unique_characters(self) -> set:
        """Extract all unique characters from the corpus."""
        return {char for word in self.corpus for char in word}

    def clean_text(self, text: str) -> List[str]:
        """
        Clean the input text by extracting words, converting to lowercase,
        and filtering based on minimum word length.
        """
        return [part.lower() for word in self.CLEAN_PATTERN.findall(text) 
                for part in word.split('-') if len(part) >= self.config.min_word_length]

    def load_corpus(self):
        """
        Load the corpus from a file or NLTK, clean it, and store as a Counter object.
        """
        file_path = self.config.corpus_dir / f'{self.corpus_name}.txt'
        if file_path.is_file():
            # Load corpus from file if it exists
            with file_path.open('r', encoding='utf-8') as file:
                for line in file:
                    self.corpus.update(self.clean_text(line))
        else:
            try:
                # Attempt to load corpus from NLTK if file doesn't exist
                nltk_corpus_name = self.corpus_name.replace('.txt', '')
                nltk.download(nltk_corpus_name, quiet=True)
                self.corpus.update(self.clean_text(' '.join(getattr(nltk.corpus, nltk_corpus_name).words())))
            except AttributeError:
                raise ValueError(f"File '{file_path}' does not exist and NLTK corpus '{nltk_corpus_name}' not found.")
            except Exception as e:
                raise RuntimeError(f"Failed to load corpus '{self.corpus_name}': {e}")

    def prepare_datasets(self):
        """
        Prepares training and testing datasets based on the chosen split type (A, B, or HAPAX).
        Updates the all_words set for comprehensive evaluation checks.
        """
        # Choose the method to split the corpus based on the specified split type
        if self.split_type == 'A':
            self._split_type_a()
        elif self.split_type == 'B':
            self._split_type_b()
        elif self.split_type == 'HAPAX':
            self._split_hapax()
        else:
            raise ValueError(f"Unknown split type: {self.split_type}")

        # Update all_words set with both training and testing words
        self.all_words = set(self.train_set)
        self.all_words.update([original_word for _, _, original_word in self.test_set])

        # Generate the formatted training list path and the corresponding models
        formatted_train_set_path = self.config.sets_dir / f'{self.corpus_name}_formatted_train_set.txt'
        self.generate_formatted_corpus(self.train_set, formatted_train_set_path)
        self.generate_models_from_corpus(formatted_train_set_path)

    def _split_hapax(self):
        """
        Split the corpus into training and testing sets based on hapax legomena.
        Hapax legomena are used as the test set, and all other tokens as the training set.
        """
        # Identify hapax legomena (words occurring exactly once in the corpus)
        hapax_legomena = {word for word, count in self.corpus.items() if count == 1}
        
        # Initialize a set to track used word-letter combinations in test set
        used_combinations = set()
        
        for word, count in self.corpus.items():
            if word in hapax_legomena:
                # Hapax legomena become part of the test set
                for _ in range(count):  # Handle the case where a hapax legomenon appears more than once due to preprocessing
                    modified_word, missing_letter, _ = self.replace_random_letter(word, used_combinations)
                    self.test_set.append((modified_word, missing_letter, word))
                    used_combinations.add((word, missing_letter))
            else:
                # All other words are added to the training set
                self.train_set.extend([word] * count)

        # Generate the formatted training list path and corresponding models
        formatted_train_set_path = self.config.sets_dir / f'{self.corpus_name}_formatted_train_set_hapax.txt'
        self.generate_formatted_corpus(self.train_set, formatted_train_set_path)
        self.generate_models_from_corpus(formatted_train_set_path)

        # Optionally log the size of the hapax legomena set
        if self.debug:
            logging.info(f"Identified {len(hapax_legomena)} unique hapax legomena for testing.")

    def _split_type_a(self):
        """
        Split Type A: Divides the list of unique words into training and testing sets based on the split configuration.
        The token frequencies associated with each word are included in the respective sets.
        """
        # Extract unique word types and shuffle them
        unique_word_types = list(self.corpus.keys())
        self.rng.shuffle(unique_word_types)

        # Calculate the split index based on the split configuration
        split_index = int(len(unique_word_types) * self.config.split_config)

        # Divide the shuffled words into training and testing sets
        training_word_types = unique_word_types[:split_index]
        testing_word_types = unique_word_types[split_index:]

        # Initialize a set to track used word-letter combinations in the test set
        used_combinations = set()

        # Assign token frequencies to the training set
        for word in training_word_types:
            self.train_set.extend([word] * self.corpus[word])

        # Assign token frequencies to the testing set, with modified words
        for word in testing_word_types:
            word_count = self.corpus[word]
            for _ in range(word_count):
                modified_word, missing_letter, _ = self.replace_random_letter(word, used_combinations)
                self.test_set.append((modified_word, missing_letter, word))
                used_combinations.add((word, missing_letter))

        # Save lists to files if debug mode is active
        if self.debug:
            self.save_list_to_file(self.train_set, f'{self.corpus_name}_train_list_a.txt')
            self.save_list_to_file(self.test_set, f'{self.corpus_name}_test_list_a.txt')

    def _split_type_b(self):
        """
        Split Type B: Randomly shuffles and divides the entire corpus into training and test lists.
        This split allows the same word tokens to appear in both training and testing data, 
        thereby maintaining word frequency balance across them. It also ensures that there 
        are no duplicate word-letter combinations in the test set, which could skew model evaluation.
        """
        # Flatten the corpus into a list of word tokens, preserving their frequency
        all_word_tokens = []
        for word, count in self.corpus.items():
            all_word_tokens.extend([word] * count)
        self.rng.shuffle(all_word_tokens)

        # Determine the split point for training and test sets based on the configured split ratio
        split_index = int(len(all_word_tokens) * self.config.split_config)
        train_tokens = all_word_tokens[:split_index]
        test_tokens = all_word_tokens[split_index:]

        # Note: Converting train_tokens to a set (training_set_words) is solely for the purpose
        # of efficiently identifying unique test tokens that do not appear in the training set.
        # This conversion does not impact the frequency of tokens in the training dataset,
        # as the training set is populated directly from train_tokens, preserving original frequencies.
        training_set_words = set(train_tokens)

        # Populate the training list, maintaining the token frequencies
        self.train_set.extend(train_tokens)

        # Track used word-letter combinations in the test set to avoid duplicates
        used_combinations = set()
        
        # Collect tokens unique to the test set for additional analysis
        unique_test_tokens = set()

        for word in test_tokens:
            modified_word, missing_letter, _ = self.replace_random_letter(word, used_combinations)
            self.test_set.append((modified_word, missing_letter, word))
            used_combinations.add((word, missing_letter))
            
            # If the word from test tokens is not found in the training set, consider it unique
            if word not in training_set_words:
                unique_test_tokens.add(word)

        # Store unique test tokens for later analysis
        self.unique_test_tokens = unique_test_tokens

        # Optionally log and save the identified unique test tokens
        if self.debug:
            self.save_list_to_file(self.train_set, f'{self.corpus_name}_train_list_b.txt')
            self.save_list_to_file(self.test_set, f'{self.corpus_name}_test_list_b.txt')

    def generate_formatted_corpus(self, data_set: List[str], formatted_corpus_path: Path) -> Path:
        """
        Generate a formatted corpus file for KenLM training.

        Args:
            data_set (list): List of words to be formatted.
            formatted_corpus_path (Path): Path where the formatted corpus will be saved.

        Returns:
            Path: Path to the generated formatted corpus file.
        """
        # Prepare a corpus file formatted for KenLM training, with each word on a new line
        formatted_text = [' '.join(word) for word in data_set]
        formatted_corpus = '\n'.join(formatted_text)

        # Save the formatted corpus to a file
        with formatted_corpus_path.open('w', encoding='utf-8') as f:
            f.write(formatted_corpus)

        return formatted_corpus_path

    def generate_models_from_corpus(self, corpus_path: Path):
        """
        Generate KenLM models for different q-gram sizes from the corpus.

        Args:
            corpus_path (Path): Path to the formatted corpus file.
        """
        # Create the directory for storing language models
        model_directory = self.config.model_dir / self.corpus_name
        model_directory.mkdir(parents=True, exist_ok=True)

        model_loaded = False
        for q in self.config.q_range:
            if q not in self.model:
                # Generate and load KenLM models for each q-gram size
                _, binary_file = build_kenlm_model(self.corpus_name, q, corpus_path, model_directory)
                if binary_file:
                    self.model[q] = kenlm.Model(str(binary_file))
                    model_loaded = True

        if model_loaded:
            logging.info(f'Model for {q}-gram loaded from {self.corpus_name}')

    def generate_and_load_models(self):
        """
        Generate and load KenLM models for the corpus if they haven't been loaded already.
        """
        # Generate and load models only if they haven't been loaded for the specified q-range
        for q in self.config.q_range:
            if q not in self.model:
                formatted_train_set_path = self.config.sets_dir / f'{self.corpus_name}_formatted_train_list.txt'
                self.generate_formatted_corpus(self.train_set, formatted_train_set_path)
                self.generate_models_from_corpus(formatted_train_set_path)

    def replace_random_letter(self, word: str, used_combinations: set) -> Tuple[str, str, str]:
        """
        Replace a random letter in the word with an underscore, avoiding previously used combinations.

        Args:
            word (str): The word to modify.
            used_combinations (set): Set of (word, index) tuples that have already been used.

        Returns:
            tuple: Modified word, missing letter, and original word.
        """
        # Identify indices of vowels and consonants in the word
        vowel_indices = [i for i, letter in enumerate(word) if letter in Letters.VOWELS.value]
        consonant_indices = [i for i, letter in enumerate(word) if letter in Letters.CONSONANTS.value]

        if not vowel_indices and not consonant_indices:
            raise ValueError(f"Unable to replace a letter in word: '{word}'.")

        # Filter indices to only those not used before
        valid_vowel_indices = [i for i in vowel_indices if (word, i) not in used_combinations]
        valid_consonant_indices = [i for i in consonant_indices if (word, i) not in used_combinations]

        # Choose from the valid indices based on the vowel replacement ratio
        letter_indices = valid_vowel_indices if self.rng.random() < self.config.vowel_replacement_ratio and valid_vowel_indices else valid_consonant_indices
        if not letter_indices:
            letter_indices = valid_vowel_indices or vowel_indices  # Fallback if no valid consonant indices

        letter_index = self.rng.choice(letter_indices)
        missing_letter = word[letter_index]
        modified_word = word[:letter_index] + '_' + word[letter_index + 1:]

        return modified_word, missing_letter, word

    def save_list_to_file(self, data_list: List, file_name: str):
        """
        Save a list of data to a file, formatting tuples appropriately.

        Args:
            data_list (list): List of data to be saved.
            file_name (str): Name of the file to save the data to.
        """
        file_path = self.config.sets_dir / file_name

        # Aggregate the data into a single string
        aggregated_data = []
        for item in data_list:
            if isinstance(item, tuple):
                # Formatting tuple as ('word', 'letter', 'original_word')
                formatted_item = f"('{item[0]}', '{item[1]}', '{item[2]}')"
                aggregated_data.append(formatted_item)
            else:
                # If it's not a tuple, just append the item
                aggregated_data.append(item)
        
        # Join all items into a single string with new lines
        aggregated_data_str = '\n'.join(aggregated_data)

        # Write the aggregated string to the file in one go
        with file_path.open('w', encoding='utf-8', buffering=8192) as file:  # 8192 bytes buffer size
            file.write(aggregated_data_str)

def analyze_test_tokens_not_in_training_performance(corpus_manager: CorpusManager, test_tokens: set) -> int:
    """
    Analyze the performance of the model on test tokens that were not present in the training data.

    Args:
        corpus_manager (CorpusManager): The corpus manager object.
        test_tokens (set): Set of unique test tokens.

    Returns:
        int: Number of analyzed tokens.
    """
    # Filter the test set to only include tokens (words) that are unique to the test set
    filtered_test_set = [
        (mod_word, miss_letter, orig_word) 
        for mod_word, miss_letter, orig_word in corpus_manager.test_set 
        if orig_word in test_tokens
    ]

    num_analyzed_tokens = len(filtered_test_set)

    if num_analyzed_tokens == 0:
        logging.info("Evaluation: No test tokens found in predictions for analysis.")
        return num_analyzed_tokens

    # Initialize an evaluation model with the filtered test set
    temp_eval_model = EvaluateModel(corpus_manager, log_initialization_details=False)
    temp_eval_model.test_set = filtered_test_set

    # Evaluate the model's predictions on the filtered test set
    evaluation_metrics, _ = temp_eval_model.evaluate_character_predictions(temp_eval_model.prediction_method)

    logging.info(f"Evaluated {num_analyzed_tokens} tokens not found in training data:")

    # Log the accuracy and validity of predictions for the top 1, 2, and 3 guesses
    for i in range(1, 4):
        accuracy = evaluation_metrics['accuracy'].get(i, 0.0)
        validity = evaluation_metrics['validity'].get(i, 0.0)
        logging.info(f"TOP{i} ACCURACY: {accuracy:.2%} | TOP{i} VALIDITY: {validity:.2%}")

    return num_analyzed_tokens

def log_evaluation_results(evaluation_metrics: dict, corpus_name: str, prediction_method_name: str):
    """
    Log the evaluation results for a given corpus and prediction method.

    Args:
        evaluation_metrics (dict): Dictionary containing accuracy and validity metrics.
        corpus_name (str): Name of the corpus.
        prediction_method_name (str): Name of the prediction method used.
    """
    logging.info(f'Evaluated with: {prediction_method_name}')
    logging.info(f'Model evaluation completed for: {corpus_name}')
    for i in range(1, 4):
        accuracy = evaluation_metrics['accuracy'].get(i, 0.0)
        validity = evaluation_metrics['validity'].get(i, 0.0)
        logging.info(f'TOP{i} ACCURACY: {accuracy:.2%} | TOP{i} VALIDITY: {validity:.2%}')

def run(corpus_name: str, config: Config, split_type: str):
    """
    Run the evaluation process for a given corpus and split type.

    Args:
        corpus_name (str): Name of the corpus to process.
        config (Config): Configuration object.
        split_type (str): Type of data split to use ('A', 'B', or 'HAPAX').
    """
    formatted_corpus_name = CorpusManager.format_corpus_name(corpus_name)
    logging.info(f'Processing {formatted_corpus_name} Corpus with split type {split_type}')
    
    # Initialize corpus manager and evaluation model
    corpus_manager = CorpusManager(formatted_corpus_name, config, split_type)
    eval_model = EvaluateModel(corpus_manager, split_type)
    prediction_method = getattr(eval_model.predictor, config.prediction_method_name)

    # Evaluate character predictions
    evaluation_metrics, predictions = eval_model.evaluate_character_predictions(prediction_method)
    log_evaluation_results(evaluation_metrics, corpus_name, prediction_method.__name__)

    # Analyze performance on unique test tokens for split type B
    if split_type == 'B':
        analyze_test_tokens_not_in_training_performance(corpus_manager, corpus_manager.unique_test_tokens)

    # Export prediction details and save summary statistics
    eval_model.export_prediction_details_to_csv(predictions, prediction_method.__name__)
    eval_model.save_summary_stats_txt(evaluation_metrics, predictions, prediction_method.__name__)

    logging.info('-' * 45)

def main():
    """
    Main function to run the evaluation process for multiple corpora and split types.
    """
    config = Config()
    config.setup_logging()
    config.create_directories()
    
    # List of corpora to process
    corpora = ['cmudict', 'brown', 'all_tokens_clmet.txt', 'all_tokens_lampeter.txt', 
               'all_tokens_openEdges.txt']
    
    # Different split types to evaluate
    split_types = ['A', 'B', 'HAPAX']
    
    # Process each corpus with each split type
    for corpus_name in corpora:
        for split_type in split_types:
            run(corpus_name, config, split_type)

if __name__ == '__main__':
    main()