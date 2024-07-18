import logging
from pathlib import Path
from typing import List, Callable
from dataclasses import dataclass, field
from corpus_class import CorpusManager
from evaluation_class import EvaluateModel

@dataclass
class Config:
    """Configuration class for setting up directories and testing parameters."""
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    seed: int = 42
    q_range: List[int] = field(default_factory=lambda: [7, 7])
    split_config: float = 0.5 # train-test split ratio
    vowel_replacement_ratio: float = 0.2 # must sum to 1 with consonant_replacement_ratio
    consonant_replacement_ratio: float = 0.8 # must sum to 1 with vowel_replacement_ratio
    min_word_length: int = 3 # minimum word length for evaluation
    prediction_method_name: str = 'context_sensitive' # method to use for prediction
    num_replacements: int = 1 # number of replacements to make: Note not all features of this code work with num_replacements > 1
    log_level: int = logging.INFO

    def __post_init__(self):
        self._set_directories()
        self.create_directories()
        self.setup_logging()

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

    def create_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in self.directories:
            directory.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """Setup logging with file and console handlers."""
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

def log_evaluation_results(evaluation_metrics: dict, corpus_name: str, prediction_method_name: str) -> None:
    """Log standard evaluation results."""
    logging.info(f'Evaluated with: {prediction_method_name}')
    logging.info(f'Model evaluation completed for: {corpus_name}')
    for i in range(1, 4):
        accuracy = evaluation_metrics['accuracy'].get(i, 0.0)
        validity = evaluation_metrics['validity'].get(i, 0.0)
        logging.info(f'TOP{i} ACCURACY: {accuracy:.2%} | TOP{i} VALIDITY: {validity:.2%}')

def process_corpus(corpus_name: str, config: Config) -> None:
    """Process a given corpus with the specified configuration and log the results."""
    try:
        formatted_corpus_name = CorpusManager.format_corpus_name(corpus_name)
        logging.info(f'Processing {formatted_corpus_name} Corpus')
        logging.info('-' * 45)

        corpus_manager = CorpusManager(formatted_corpus_name, config)
        CorpusManager.add_to_global_corpus(corpus_manager.corpus)

        eval_model = EvaluateModel(corpus_manager)
        prediction_method: Callable = getattr(eval_model.predictor, config.prediction_method_name)

        evaluation_metrics, predictions = eval_model.evaluate_character_predictions(prediction_method)

        log_evaluation_results(evaluation_metrics, corpus_name, prediction_method.__name__)

        eval_model.export_prediction_details_to_csv(predictions, prediction_method.__name__)
        eval_model.save_summary_stats_txt(evaluation_metrics, predictions, prediction_method.__name__)

        logging.info('-' * 45)
    except Exception as e:
        logging.error(f"Error processing corpus {corpus_name}: {e}", exc_info=True)

def main() -> None:
    """Main function to setup configuration and process corpora."""
    config = Config()

    corpora: List[str] = [
        'cmudict', 'brown', 'sorted_tokens_clmet.txt', 
        'sorted_tokens_lampeter.txt', 'sorted_tokens_openEdges.txt'
    ]
    
    for corpus_name in corpora:
        process_corpus(corpus_name, config)

if __name__ == '__main__':
    main()