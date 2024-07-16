import csv
import logging
from predictions_class import Predictions

class EvaluateModel:
    """
    A class for evaluating character prediction models in a language processing context.
    
    This class handles the evaluation of predictions for missing characters in words,
    computing metrics such as accuracy and validity, and saving detailed results.
    It's designed to work with various prediction methods and corpus configurations.
    """

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
        self.corpus = corpus_manager.corpus
        self.train_set = corpus_manager.train_set
        self.test_set = corpus_manager.test_set
        self.all_words = corpus_manager.all_words
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

    def compute_metrics(self, predictions) -> dict:
        """
        Compute accuracy and validity metrics for the predictions.
        
        Args:
            predictions: List of prediction results for each test word.
        
        Returns:
            A dictionary containing accuracy and validity metrics for top 1, 2, and 3 predictions.
        """
        accuracy_counts = {1: 0, 2: 0, 3: 0}
        validity_counts = {1: 0, 2: 0, 3: 0}
        total_test_words = len(self.test_set)

        for modified_word, missing_letters, _, all_predictions, _ in predictions:
            # Validate prediction format to ensure consistency
            if not all(isinstance(pred, tuple) and len(pred) == 2 for pred in all_predictions):
                logging.error(f'Invalid prediction format: {all_predictions}')
                continue

            # Find the rank of the first correct prediction (accuracy)
            correct_rank = next((rank for rank, (predicted_letter, _) in enumerate(all_predictions, start=1)
                                if predicted_letter in missing_letters), None)

            # Update accuracy counts for all ranks up to and including the correct rank
            if correct_rank:
                for rank in range(correct_rank, 4):
                    accuracy_counts[rank] += 1

            # Check validity: whether predictions result in valid words
            valid_word_checked = [False] * 3
            for rank, (predicted_letter, _) in enumerate(all_predictions[:3], start=1):
                if not valid_word_checked[rank-1]:
                    reconstructed_word = modified_word.replace('_', predicted_letter, 1)
                    if reconstructed_word in self.all_words:
                        # Mark as valid for this and all higher ranks
                        for i in range(rank, 4):
                            if not valid_word_checked[i-1]:
                                validity_counts[i] += 1
                                valid_word_checked[i-1] = True

        # Calculate final metrics as percentages for easy interpretation
        total_accuracy = {k: accuracy_counts[k] / total_test_words for k in accuracy_counts}
        total_validity = {k: validity_counts[k] / total_test_words for k in validity_counts}

        return {'accuracy': total_accuracy, 'validity': total_validity, 'total_words': total_test_words}

    def evaluate_character_predictions(self, prediction_method) -> tuple[dict, list]:
        """
        Evaluate character predictions for the test set.
        
        Args:
            prediction_method: The method to use for making predictions.
        
        Returns:
            A tuple containing evaluation metrics and detailed prediction results.
        """
        predictions = []

        for modified_word, target_letters, original_word in self.test_set:
            try:
                # Get predictions for the modified word using the specified method
                all_predictions = prediction_method(modified_word)
                
                # Validate prediction format to ensure consistency
                if not isinstance(all_predictions, list) or not all(isinstance(pred, tuple) and len(pred) == 2 for pred in all_predictions):
                    logging.error(f'Unexpected prediction format for {modified_word}: {all_predictions}')
                    continue

                # Find the rank of the correct prediction (if any)
                correct_letter_rank = next((rank for rank, (retrieved_letter, _) in enumerate(all_predictions, start=1)
                                            if retrieved_letter in target_letters), None)

                # Store prediction details for later analysis
                predictions.append((modified_word, target_letters, original_word, all_predictions[:3], correct_letter_rank))
            except Exception as e:
                logging.error(f"Error predicting for {modified_word}: {e}", exc_info=True)

        # Compute overall metrics based on all predictions
        evaluation_metrics = self.compute_metrics(predictions)
        return evaluation_metrics, predictions

    def save_summary_stats_txt(self, evaluation_metrics, predictions, prediction_method_name):
        """
        Save a detailed summary of evaluation results to a text file.
        
        Args:
            evaluation_metrics: Dictionary containing accuracy and validity metrics.
            predictions: List of detailed prediction results.
            prediction_method_name: Name of the prediction method used.
        """
        output_file_path = self.config.text_dir / f'{self.corpus_name}_predictions.txt'
        try:
            with output_file_path.open('w', encoding='utf-8') as file:
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

                # Write detailed prediction results for each test word
                for mod_word, miss_letters, orig_word, top_preds, cor_letter_rank in predictions:
                    file.write(f'Test Word: {mod_word}, Correct Letters: {",".join(miss_letters)}\n')
                    file.write(f'Correct Letter Rank: {cor_letter_rank}\n')

                    for rank, (pred_letter, confidence) in enumerate(top_preds, start=1):
                        reconstructed_word = mod_word.replace('_', pred_letter)
                        is_valid_word = reconstructed_word in self.all_words

                        file.write(f"Rank {rank}: '{pred_letter}' (Confidence: {confidence:.8f}), "
                                   f"Valid: {is_valid_word}\n")

                    file.write('\n')
        except Exception as e:
            logging.error(f"Error saving summary stats to {output_file_path}: {e}", exc_info=True)

    def export_prediction_details_to_csv(self, predictions, prediction_method_name):
        """
        Export detailed prediction results to a CSV file for further analysis.
        
        Args:
            predictions: List of detailed prediction results.
            prediction_method_name: Name of the prediction method used.
        """
        # Construct CSV file path with relevant details for easy identification
        split_type_str = f"_{self.split_type}" if self.split_type else ""
        csv_file_path = self.config.csv_dir / (
            f'{self.corpus_name}_{prediction_method_name}{split_type_str}_split'
            f'{self.config.split_config}_qrange{self.config.q_range[0]}-'
            f'{self.config.q_range[1]}_prediction.csv'
        )
        try:
            with csv_file_path.open('w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)

                # Write CSV header with detailed column descriptions
                writer.writerow([
                    'Tested_Word', 'Original_Word', 'Correct_Letter(s)', 
                    'Top1_Predicted_Letter', 'Top1_Confidence', 'Top1_Is_Valid', 'Top1_Is_Accurate',
                    'Top2_Predicted_Letter', 'Top2_Confidence', 'Top2_Is_Valid', 'Top2_Is_Accurate',
                    'Top3_Predicted_Letter', 'Top3_Confidence', 'Top3_Is_Valid', 'Top3_Is_Accurate',
                    'Correct_Letter_Rank', 'In_Training_Set'
                ])

                # Create a set of training words for quick lookup during validation
                training_words_set = set(self.train_set)
                
                # Write prediction details for each test word
                for mod_word, miss_letters, orig_word, top_preds, cor_letter_rank in predictions:
                    row = [mod_word, orig_word, ','.join(miss_letters)]

                    for predicted_letter, confidence in top_preds:
                        reconstructed_word = mod_word.replace('_', predicted_letter)
                        is_valid = 1 if reconstructed_word in self.all_words else 0
                        is_accurate = 1 if predicted_letter in miss_letters else 0

                        row.extend([predicted_letter, confidence, is_valid, is_accurate])

                    row.append(cor_letter_rank)
                    row.append(1 if orig_word in training_words_set else 0)

                    writer.writerow(row)
        except Exception as e:
            logging.error(f"Error exporting prediction details to CSV {csv_file_path}: {e}", exc_info=True)