import numpy as np
import logging
import gzip
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures 
from predictions_class import Predictions

class EvaluateModel:
    def __init__(self, corpus_manager, split_type=None, log_initialization_details=True):
        """
        Initialize the EvaluateModel with necessary data and configurations.
        
        Args:
            corpus_manager: An object managing the corpus data and model.
            split_type: Type of data split (e.g., 'random', 'chronological'), if applicable.
            log_initialization_details: Whether to log initialization details.
        """
        # Store corpus manager and extract relevant attributes for easy access
        self.corpus_manager = corpus_manager
        self.corpus_name = corpus_manager.corpus_name
        self.config = corpus_manager.config
        self.model = corpus_manager.model

        # Convert lists to NumPy arrays for faster operations
        self.corpus = np.array(corpus_manager.corpus)  
        self.train_set = np.array(corpus_manager.train_set)  
        self.test_set = np.array(corpus_manager.test_set)  
        self.all_words = np.array(corpus_manager.all_words)  
        self.split_type = split_type

        # Extract unique characters from the corpus for character-level analysis
        unique_characters = corpus_manager.extract_unique_characters()
        self.unique_character_count = len(unique_characters)

        # Initialize prediction class with model, q-gram range, and character set
        self.q_range = range(self.config.q_range[0], self.config.q_range[1] + 1)
        self.predictor = Predictions(self.model, self.q_range, unique_characters)

        # Log initialization details if requested, useful for experiment tracking
        if log_initialization_details:
            self.log_initialization_details()

        # Set up the prediction method to be used based on configuration
        self.prediction_method = self.get_prediction_method()

    def get_prediction_method(self):
        """
        Retrieve the appropriate prediction method based on configuration.
        
        Returns:
            A function representing the selected prediction method.
        """
        # Dictionary mapping prediction method names to their corresponding functions
        prediction_methods = {
            'context_sensitive': self.predictor.context_sensitive,
            'context_no_boundary': self.predictor.context_no_boundary,
            'base_prediction': self.predictor.base_prediction
        }
        # Default to context_sensitive if specified method not found
        return prediction_methods.get(self.config.prediction_method_name, self.predictor.context_sensitive)

    def log_initialization_details(self):
        """
        Log important details about the model initialization and configuration.
        This aids in experiment tracking and reproducibility.
        """
        # Log various configuration parameters and dataset statistics
        logging.info(f'Language Model for {self.corpus_name} initialized with:')
        logging.info(f'Seed: {self.config.seed}')
        logging.info(f'Q-gram Range: {self.config.q_range}')
        logging.info(f'Train-Test Split Configuration: {self.config.split_config}')
        logging.info(f'Training Set Size: {len(self.train_set)}')
        logging.info(f'Testing Set Size: {len(self.test_set)}')
        logging.info(f'Vowel Replacement Ratio: {self.config.vowel_replacement_ratio}')
        logging.info(f'Consonant Replacement Ratio: {self.config.consonant_replacement_ratio}')
        logging.info(f'Unique Character Count: {self.unique_character_count}')
        logging.info(f'Minimum Word Length: {self.config.min_word_length}')
        logging.info(f'Number of Replacements per Word: {self.config.num_replacements}')

    def compute_metrics(self, predictions) -> dict[str, dict[int, float]]:
        """
        Compute accuracy and validity metrics for the predictions.

        Args:
            predictions: List of prediction results for each test word.

        Returns:
            A dictionary containing accuracy and validity metrics for top 1, 2, and 3 predictions.
        """
        total_test_words = len(self.test_set)
        
        # Convert predictions to a NumPy array for faster processing
        pred_array = np.array(predictions, dtype=object)
        
        # Initialize counters for accuracy and validity
        accuracy_counts = np.zeros(3, dtype=int)
        validity_counts = np.zeros(3, dtype=int)
        
        for modified_word, missing_letters, _, all_predictions, _ in pred_array:
            # Determine the rank of the first correct prediction
            correct_rank = np.argmax([pred[0] in missing_letters for pred in all_predictions]) + 1 if any([pred[0] in missing_letters for pred in all_predictions]) else None
            
            if correct_rank:
                # Increment accuracy counts for all ranks up to the correct rank
                accuracy_counts[correct_rank-1:] += 1
            
            # Check the validity of the top 3 predictions
            for rank, (predicted_letter, _) in enumerate(all_predictions[:3], start=1):
                reconstructed_word = modified_word.replace('_', predicted_letter, 1)
                if reconstructed_word in self.all_words:
                    validity_counts[rank-1:] += 1
                    break

        # Calculate accuracy and validity as percentages
        total_accuracy = accuracy_counts / total_test_words
        total_validity = validity_counts / total_test_words
        
        return {
            'accuracy': {k+1: v for k, v in enumerate(total_accuracy)},
            'validity': {k+1: v for k, v in enumerate(total_validity)},
            'total_words': total_test_words
        }

    def evaluate_character_predictions(self, prediction_method) -> tuple[dict, list]:
        """
        Evaluate the model's character-level predictions using the specified method.
        """
        predictions = []
        
        with ThreadPoolExecutor() as executor:
            future_to_word = {executor.submit(self._predict_word, prediction_method, modified_word, target_letters, original_word): (modified_word, target_letters, original_word) for modified_word, target_letters, original_word in self.test_set}
            for future in concurrent.futures.as_completed(future_to_word):
                try:
                    predictions.append(future.result())
                except Exception as exc:
                    modified_word, target_letters, original_word = future_to_word[future]
                    logging.error(f"Error predicting for {modified_word}: {exc}", exc_info=True)

        evaluation_metrics = self.compute_metrics(predictions)
        return evaluation_metrics, predictions

    def _predict_word(self, prediction_method, modified_word, target_letters, original_word):
        """
        Predict the missing letters in a word using the specified prediction method.
        """
        try:
            all_predictions = prediction_method(modified_word)
            if not isinstance(all_predictions, list) or not all(isinstance(pred, tuple) and len(pred) == 2 for pred in all_predictions):
                logging.error(f'Unexpected prediction format for {modified_word}: {all_predictions}')
                return None

            correct_letter_rank = next((rank for rank, (retrieved_letter, _) in enumerate(all_predictions, start=1)
                                        if retrieved_letter in target_letters), None)

            return (modified_word, target_letters, original_word, all_predictions[:3], correct_letter_rank)
        except Exception as e:
            logging.error(f"Error predicting for {modified_word}: {e}", exc_info=True)
            return None

    def save_summary_stats_txt(self, evaluation_metrics, predictions, prediction_method_name):
        """
        Save a detailed summary of evaluation results to a text file.
        
        Args:
            evaluation_metrics: Dictionary containing accuracy and validity metrics.
            predictions: List of detailed prediction results.
            prediction_method_name: Name of the prediction method used.
        """
        # Define the output file path
        output_file_path = self.config.text_dir / f'{self.corpus_name}_predictions.txt'
        
        try:
            # Open the file with a large buffer size for efficient writing
            with output_file_path.open('w', encoding='utf-8', buffering=1024*1024) as file:
                # Write general information about the evaluation
                file.write(f'Prediction Method: {prediction_method_name}\n')
                file.write(f'Unique Character Count: {self.unique_character_count}\n\n')

                # Write accuracy and validity metrics for easy reference
                accuracy = evaluation_metrics['accuracy']
                validity = evaluation_metrics['validity']
                for i in range(1, 4):
                    file.write(f'TOP{i} ACCURACY: {accuracy[i]:.2%}\n')
                    file.write(f'TOP{i} VALIDITY: {validity[i]:.2%}\n')
                file.write('\n')

                # Write dataset information for context
                file.write(f'Train Size: {len(self.train_set)}, Test Size: {len(self.test_set)}\n')
                file.write(f'Vowel Ratio: {self.config.vowel_replacement_ratio}, '
                        f'Consonant Ratio: {self.config.consonant_replacement_ratio}\n\n')

                # Convert predictions to a NumPy array for faster processing
                pred_array = np.array(predictions, dtype=object)

                # Vectorize the 'in' operation for faster lookup of valid words
                is_valid_word_vec = np.vectorize(lambda w: w in self.all_words)

                # Process each prediction
                for mod_word, miss_letters, orig_word, top_preds, cor_letter_rank in pred_array:
                    file.write(f'Test Word: {mod_word}, Correct Letters: {",".join(miss_letters)}\n')
                    file.write(f'Correct Letter Rank: {cor_letter_rank}\n')

                    # Prepare reconstructed words for validity check
                    reconstructed_words = [mod_word.replace('_', pred[0]) for pred in top_preds]
                    is_valid_words = is_valid_word_vec(reconstructed_words)

                    # Write top predictions and their validity
                    for rank, ((pred_letter, confidence), is_valid) in enumerate(zip(top_preds, is_valid_words), start=1):
                        file.write(f"Rank {rank}: '{pred_letter}' (Confidence: {confidence:.8f}), Valid: {is_valid}\n")

                    file.write('\n')
        
        except Exception as e:
            # Log any errors that occur during the file writing process
            logging.error(f"Error saving summary stats to {output_file_path}: {e}", exc_info=True)

    def export_prediction_details_to_csv(self, predictions, prediction_method_name):
        """
        Export detailed prediction results to a CSV file for further analysis.
        
        Args:
            predictions: List of detailed prediction results.
            prediction_method_name: Name of the prediction method used.
        """
        # Construct paths for the intermediate gzip file and the final CSV file
        split_type_str = f"_{self.split_type}" if self.split_type else ""
        gzip_file_path = self.config.csv_dir / (
            f'{self.corpus_name}_{prediction_method_name}{split_type_str}_split'
            f'{self.config.split_config}_qrange{self.config.q_range[0]}-'
            f'{self.config.q_range[1]}_prediction.csv.gz'
        )
        csv_file_path = self.config.csv_dir / (
            f'{self.corpus_name}_{prediction_method_name}{split_type_str}_split'
            f'{self.config.split_config}_qrange{self.config.q_range[0]}-'
            f'{self.config.q_range[1]}_prediction.csv'
        )
        
        try:
            # Convert predictions to a NumPy array for faster processing
            pred_array = np.array(predictions, dtype=object)

            # Create sets for faster lookup
            all_words_set = set(self.all_words)
            training_words_set = set(self.train_set)

            # Vectorize operations for speedup
            is_valid_word_vec = np.vectorize(lambda w: w in all_words_set)
            is_in_training_set_vec = np.vectorize(lambda w: w in training_words_set)

            # Extract data from prediction array for processing
            mod_words, miss_letters, orig_words, top_preds, cor_letter_ranks = pred_array.T

            # Prepare reconstructed words and validity checks
            reconstructed_words = np.array([[mod.replace('_', pred[0]) for pred in preds] for mod, preds in zip(mod_words, top_preds)])
            is_valid = is_valid_word_vec(reconstructed_words)
            is_accurate = np.array([[pred[0] in miss for pred in preds] for miss, preds in zip(miss_letters, top_preds)])
            in_training_set = is_in_training_set_vec(orig_words)

            # Prepare data as a structured NumPy array
            dtype = [('Tested_Word', 'U50'), ('Original_Word', 'U50'), ('Correct_Letters', 'U10')]
            for i in range(1, 4):
                dtype.extend([
                    (f'Top{i}_Predicted_Letter', 'U1'),
                    (f'Top{i}_Confidence', 'f8'),
                    (f'Top{i}_Is_Valid', 'i4'),
                    (f'Top{i}_Is_Accurate', 'i4')
                ])
            dtype.extend([('Correct_Letter_Rank', 'i4'), ('In_Training_Set', 'i4')])

            data = np.empty(len(pred_array), dtype=dtype)
            data['Tested_Word'] = mod_words
            data['Original_Word'] = orig_words
            data['Correct_Letters'] = [','.join(miss) for miss in miss_letters]
            
            for i in range(3):
                data[f'Top{i+1}_Predicted_Letter'] = [preds[i][0] if i < len(preds) else '' for preds in top_preds]
                data[f'Top{i+1}_Confidence'] = [preds[i][1] if i < len(preds) else np.nan for preds in top_preds]
                data[f'Top{i+1}_Is_Valid'] = is_valid[:, i]
                data[f'Top{i+1}_Is_Accurate'] = is_accurate[:, i]
            
            data['Correct_Letter_Rank'] = cor_letter_ranks
            data['In_Training_Set'] = in_training_set

            # Write to compressed CSV file using gzip for on-the-fly compression
            with gzip.open(gzip_file_path, 'wt', newline='', encoding='utf-8') as f:
                # Use numpy.savetxt for fast CSV writing
                np.savetxt(f, data, delimiter=',', fmt='%s', header=','.join(dtype[0] for dtype in data.dtype.descr))

            # Extract gzip file to regular CSV file
            with gzip.open(gzip_file_path, 'rt', encoding='utf-8') as gz_file, csv_file_path.open('w', newline='', encoding='utf-8') as csv_file:
                csv_file.write(gz_file.read())

            # Delete the intermediate gzip file
            gzip_file_path.unlink()

        except Exception as e:
            logging.error(f"Error exporting prediction details to CSV {csv_file_path}: {e}", exc_info=True)
