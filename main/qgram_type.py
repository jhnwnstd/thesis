import logging
from pathlib import Path
from corpus_class import CorpusManager
from evaluation_class import EvaluateModel

class Config:
    """Configuration class for setting up directories and default parameters."""

    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir if base_dir else __file__).parent
        self._set_directories()
        self._set_values()
        self.create_directories()

    def _set_directories(self):
        """Setup various directories needed for the application."""
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

    def _set_values(self):
        """Values for testing."""
        self.seed = 42
        self.q_range = [7, 7]
        self.split_config = 0.5
        self.vowel_replacement_ratio = 0.2
        self.consonant_replacement_ratio = 0.8
        self.min_word_length = 3
        self.prediction_method_name = 'context_sensitive'
        self.num_replacements = 1  # Default number of replacements
        self.log_level = logging.INFO

    def setup_logging(self):
        """Setup logging with file and console handlers."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logfile = self.log_dir / 'logfile.log'
        logging.basicConfig(level=self.log_level,
                            format='%(asctime)s - %(message)s',
                            handlers=[
                                logging.FileHandler(logfile, mode='a'),
                                logging.StreamHandler()
                            ])

    def create_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in self.directories:
            directory.mkdir(parents=True, exist_ok=True)

def log_evaluation_results(evaluation_metrics, corpus_name, prediction_method_name):
    """Helper function to log standard evaluation results."""
    logging.info(f'Evaluated with: {prediction_method_name}')
    logging.info(f'Model evaluation completed for: {corpus_name}')
    for i in range(1, 4):
        accuracy = evaluation_metrics['accuracy'].get(i, 0.0)
        validity = evaluation_metrics['validity'].get(i, 0.0)
        logging.info(f'TOP{i} ACCURACY: {accuracy:.2%} | TOP{i} VALIDITY: {validity:.2%}')

def run(corpus_name, config):
    """
    Process a given corpus with the specified configuration and log the results.
    """
    try:
        formatted_corpus_name = CorpusManager.format_corpus_name(corpus_name)
        logging.info(f'Processing {formatted_corpus_name} Corpus')
        logging.info('-' * 45)

        corpus_manager = CorpusManager(formatted_corpus_name, config)
        CorpusManager.add_to_global_corpus(corpus_manager.corpus)

        eval_model = EvaluateModel(corpus_manager)
        prediction_method = getattr(eval_model.predictor, config.prediction_method_name)

        evaluation_metrics, predictions = eval_model.evaluate_character_predictions(prediction_method)

        # Log evaluation results using the new function
        log_evaluation_results(evaluation_metrics, corpus_name, prediction_method.__name__)

        # Export details and summary
        eval_model.export_prediction_details_to_csv(predictions, prediction_method.__name__)
        eval_model.save_summary_stats_txt(evaluation_metrics, predictions, prediction_method.__name__)

        logging.info('-' * 45)
    except Exception as e:
        logging.error(f"Error processing corpus {corpus_name}: {e}", exc_info=True)

def main():
    """Setup logging and create necessary directories."""
    config = Config()
    config.setup_logging()

    # Iterating over each corpus for processing
    corpora = ['cmudict', 'brown', 'sorted_tokens_clmet.txt', 'sorted_tokens_lampeter.txt', 'sorted_tokens_openEdges.txt']
    for corpus_name in corpora:
        run(corpus_name, config)

if __name__ == '__main__':
    main()
